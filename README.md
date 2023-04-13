
# sc-learn

This library provides a machine learning and statistical learning toolbox for
single cell and builk rna-seq compatible with scikit-learn API.

### Notes:

the aim of this work is to provide a set of state-of-art bioinformatic tools
so that they can be integrated seamlessy in python machine learning pipelines.

i'm not going to build new implementation of tools that are already available
(eg. UMAP or TSNE).
Some of the implementation are adapted from other libraries ( mainly scanpy ) that
work very well but have it own API.


### currently implemented estimators:

- clustering:
  - PhenoClust: implementation of PhenoGraph, a clustering algorithm for single cell
  rna-seq data, implemented with networkx and scikit-learn.
- gene selection, feature selection algorithms implemented with SelectorMixin:
  - HVGeneSelector: select only higly variable genes. reproduce seurat implementation.
  the attribute 'gene_stat_' can be accessed to have a report about higly variable genes
  analysis. partially adapted from scanpy implementation.
  - LowQualityGeneFIlter: filter out low quality genes
- normalization, preprocessing algorithms implemented with TransformerMixin:
  - PearsonResidualsNormalizer: analytic pearson residuals normalization.
  partially adapted from scanpy normalization
  - TMMNormalizer: tmm normalization
  - SequencingDepthNormalizer: sequencing depth normalizer
  - GfIcfNormalizer: gene-frequency inverse-cell frequency implementation

### TODO:

- differentially expressed genes selector
- extend documentation
