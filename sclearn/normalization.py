import numpy as np
from scipy.sparse import issparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


class PearsonResidualsNormalizer(TransformerMixin, BaseEstimator):
    """
    Applies analytic Pearson residual normalization.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Adapted from scanpy implementation
    """

    def __init__(self, theta=100, clip=None):
        assert (theta > 0)  # check theta
        assert (clip is None or clip >= 0)
        self.theta = theta
        self.clip = clip

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        scdata = np.array(X, dtype=int)  # cast to numpy array
        mask = scdata.sum(axis=0) > 0  # ignore all 0 columns
        X = scdata[:, mask]

        # prepare clipping
        if self.clip is None:
            n = X.shape[0]
            clip = np.sqrt(n)
        else:
            clip = self.clip

        if issparse(X):

            sums_genes = np.sum(X, axis=0)
            sums_cells = np.sum(X, axis=1)
            sum_total = np.sum(sums_genes).squeeze()
        else:
            sums_genes = np.sum(X, axis=0, keepdims=True)
            sums_cells = np.sum(X, axis=1, keepdims=True)
            sum_total = np.sum(sums_genes)

        mu = np.array(sums_cells @ sums_genes / sum_total)
        diff = np.array(X - mu)
        residuals = diff / np.sqrt(mu + mu ** 2 / self.theta)

        # clip
        scdata[:, mask] = np.clip(residuals, a_min=-clip, a_max=clip)
        return scdata

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)


class TMMNormalizer(TransformerMixin, BaseEstimator):
    """
    Applies TMM Normalization like in edgeR package.

    To compute scale factors a reference sample is choosen as the one nearest to the specified
    percentile of data ( ref_percentile params). GeneWise Logfold change matrix (M) and
    absolute expression level (A) is computed and mean-trimmed according to trim_m and trim_a params.

    if seq_depth is set to True a sequencing depth normalization is also done. total numer of reads
    per sample is set as the median of the sequencing depth of the train set.

    Based on conorm implementation, but designed to scale with data size ( conorm implementation can't
    handle large datasets )
    """

    def __init__(self, trim_m=.3, trim_a=.05, ref_percentile=75, seq_depth=True):
        self.trim_m = trim_m
        self.trim_a = trim_a
        self.ref_percentile = ref_percentile
        self.scale_factor = None
        self.total_counts = None

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y=None):
        self.total_counts = np.median(np.sum(X,axis=1))
        self.scale_factor = self.__compute_scale_factors(X)
        return self

    def transform(self, X, y=None):
        X = normalize(X, norm='l1') * self.total_counts
        X /= self.scale_factor
        return X

    def __compute_scale_factors(self, readcounts):
        """
        compute scale factor
        :param readcounts: expression matrix of shape (n_sample,n_genes)
        :return: ndarray of shape (n_sample,1) with the scale factors of each sample
        """

        def scale_factor(x, ref):
            """
            compute scale factor for a sample
            :param x: sample
            :param ref:  reference sample
            :return:  scale factor for the sample
            """
            vsize = x.shape[0] // 2
            mask = x[vsize:].astype(bool)
            x = x[:vsize]

            norm_x = x / np.nansum(x)
            norm_ref = ref / np.nansum(ref)
            log_sample = np.log2(norm_x)
            log_ref = np.log2(ref)
            m = log_sample - log_ref
            a = (log_sample + log_ref) / 2

            perc_m = np.nanquantile(m, [self.trim_m, 1 - self.trim_m], method='nearest')
            perc_a = np.nanquantile(a, [self.trim_a, 1 - self.trim_a], method='nearest')

            mask |= (m < perc_m[0]) | (m > perc_m[1])
            mask |= (a < perc_a[0]) | (a > perc_a[1])

            w = ((1 - norm_x) / x) + ((1 - norm_ref) / ref)
            w = 1 / w

            w[mask] = 0
            m[mask] = 0
            w /= w.sum()
            return np.sum(w * m)

        readcounts = np.array(readcounts, dtype=float)
        q_expr = np.apply_along_axis(lambda x: np.percentile(x[np.any(x != 0)], self.ref_percentile),
                                     axis=1, arr=readcounts)
        iref = np.argmin(np.abs(q_expr - q_expr.mean()))
        refsample = readcounts[iref, :]

        f = readcounts == 0
        f[:, f[iref]] = True
        readcounts[f] = np.nan

        funcin = np.concatenate((readcounts, f), axis=1)
        sf = np.apply_along_axis(lambda x: scale_factor(x, refsample), axis=1, arr=funcin)
        sf -= sf.mean()
        return np.exp2(sf)


class SequencingDepthNormalizer(TransformerMixin, BaseEstimator):
    """
    Sequencing depth normalization
    'total' parameter can be used to set the total number of counts per sample
    in the transformed matrix. default is 1e6, that is equal to a cpm normalization.
    if total is set to 'mean' or 'median', it will be estimated on the train set in the
    fit function.
    a log transformation can also be done if log=True
    """
    def __init__(self, total=1e6, log=False):
        self.total = total
        self.log = log

    def fit(self, X, y=None):
        if self.total == 'mean':
            self.total = np.sum(X, axis=1).mean()
        elif self.total == 'median':
            self.total = np.median(np.sum(X, axis=1))
        return self

    def transform(self, X, y=None):
        X = normalize(X, norm='l1') * self.total
        if self.log:
            return np.log(X + 1)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class GfIcfNormalizer(TransformerMixin, BaseEstimator):
    """
    Applies a gene-frequency inverse-cell-frequency normalization.
    It's basicaly just a wrapper for vanilla sklearn's tfidf transformer
    but output is converted to dense ndarray for easy integration with
    pipelines.
    """
    def __init__(self):
        self.tfidf = TfidfTransformer()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

    def transform(self, X, y=None):
        return self.tfidf.transform(X).toarray()

    def fit_transform(self, X, y=None, **fit_params):
        return self.tfidf.fit_transform(X).toarray()
