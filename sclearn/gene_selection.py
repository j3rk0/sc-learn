from abc import ABCMeta

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from scipy.stats import wilcoxon, rankdata


"""
feature selector use the sklearn's SelectorMixin implementation.
read sklearn documentation for more infos
"""


class HVGeneSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Select only highly variable genes. It reproduces seurat's implementation
    partially adapted from scanpy implementation
    """

    def __init__(self, min_disp=0.5,
                 max_disp=np.inf,
                 min_mean=0.0125,
                 max_mean=3,
                 n_top_genes=None,
                 n_bins=20):

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.n_top_genes = n_top_genes
        self.n_bins = n_bins
        self.mask_ = None
        self.gene_stats_ = None

    def fit(self, X, y=None):
        X = np.array(X)
        mean = np.mean(X, axis=0, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=0, dtype=np.float64)
        var = mean_sq - mean ** 2
        # compute variance
        var *= X.shape[0] / (X.shape[0] - 1)
        # now actually compute the dispersion
        mean[mean == 0] = 1e-12  # set entries equal to zero to small value
        dispersion = var / mean
        # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)

        # all the following quantities are "per-gene" here
        df = pd.DataFrame()
        df['means'] = mean
        df['dispersions'] = dispersion

        df['mean_bin'] = pd.cut(df['means'], bins=self.n_bins)
        disp_grouped = df.groupby('mean_bin')['dispersions']
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()

        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[
            one_gene_per_bin.values
        ].values
        disp_mean_bin[one_gene_per_bin.values] = 0

        # actually do the normalization
        df['dispersions_norm'] = (df['dispersions'].values  # use values here as index differs
                                  - disp_mean_bin[df['mean_bin'].values].values
                                  ) / disp_std_bin[df['mean_bin'].values].values

        dispersion_norm = df['dispersions_norm'].values
        if self.n_top_genes is not None:
            dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
            dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
            if self.n_top_genes > X.shape[1]:
                self.n_top_genes = X.shape[1]
            if self.n_top_genes > dispersion_norm.size:
                self.n_top_genes = dispersion_norm.size
            disp_cut_off = dispersion_norm[self.n_top_genes - 1]
            gene_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
        else:
            dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
            gene_subset = np.logical_and.reduce(
                (
                    mean > self.min_mean,
                    mean < self.max_mean,
                    dispersion_norm > self.min_disp,
                    dispersion_norm < self.max_disp,
                )
            )

        df['differentialy_expressed'] = gene_subset
        self.gene_stats_ = df
        self.mask_ = gene_subset
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def _get_support_mask(self):
        return self.mask_


class LowQualityGeneFilter(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    """
    Perform a filtering on genes.

    """

    def __init__(self, min_cell_per_gene_pct=15,
                 min_nonz_expr=1.12,
                 min_detect_gene=.05):
        """

        :param min_cell_per_gene_pct:
        :param min_nonz_expr:
        :param min_detect_gene:
        """
        self.min_cell_per_gene_pct = min_cell_per_gene_pct
        self.min_nonz_expr = min_nonz_expr
        self.min_detect_gene = min_detect_gene
        self.mask_ = None

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        cell_count_cutoff = np.log10(self.min_cell_per_gene_pct)  # genes detected in less than 15% will be excluded
        nonz_mean_cutoff = np.log10(
            self.min_nonz_expr)  # if cutoff2 < gene < cutoff, select only genes with nonzero mean expression > 1.12
        cell_count_cutoff2 = np.log10(
            X.shape[0] * self.min_detect_gene + 1e-7)  # genes detected in at least this will be included
        cells_per_gene = (X > 0).sum(axis=0) + 1e-7  # computing count of cells per gene
        nonz_mean = X.sum(axis=0) / cells_per_gene + 1e-7  # computing mean expression

        # switch to log10
        cells_per_gene = np.log10(cells_per_gene)
        nonz_mean = np.log10(nonz_mean)

        # creating filter as a boolean array
        self.mask_ = np.array(cells_per_gene > cell_count_cutoff2) | (
                np.array(cells_per_gene > cell_count_cutoff)
                & np.array(nonz_mean > nonz_mean_cutoff))

    def _get_support_mask(self):
        return self.mask_


class DEGeneSelector(SelectorMixin, BaseEstimator, metaclass=ABCMeta):
    # TODO
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y):
        assert (y is not None)
        X = np.array(X)
        groups = np.unique(y)

        x0 = X[y == groups[0], :]
        x1 = X[y == groups[1], :]

        # compute adjusted pval
        statistic, p_val = self.test(x0, x1)
        ranked_p_val = rankdata(p_val)
        qval = p_val * len(p_val) / ranked_p_val
        qval[qval > 1] = 1
        qval = -np.log10(qval)

        log2fc = np.log2(np.mean(x0, axis=0)) - np.log2(np.mean(x1, axis=0))

        self.gene_stats_ = np.array([log2fc, qval])
        logfc = np.reshape(log2fc, -1)
        logfc = np.clip(logfc, -self.log2_fc_threshold, self.log2_fc_threshold, logfc)
        return self

    def _get_support_mask(self):
        return self.mask_

    def __init__(self, test=wilcoxon, log2_fc_threshold=10, alfa=.05, min_fc=1):
        self.test = test
        self.log2_fc_threshold = log2_fc_threshold
        self.alfa = alfa
        self.min_fc = min_fc

        self.mask_ = None
        self.gene_stats_ = None
