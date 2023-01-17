# WGCNA
+ 5 WGCNA steps:
  + Construct gene co-expression network
    + calculate similarity matrix
    + calculate adjacency matrix
      + decide which type of network (unsigned, signed, signed hybrid)
    + set soft-threshold beta
      + scale independence OR mean connectivity
    + create topological overlap matrix (TOM) from adj. matrix
  + Identify modules
    + form hierarchical clusterin/dendrogram using matrix value with pairwise comparison
    + identify nested clusters (also outlier detection)
  + relate modules to external information
    + e.g. variable like weight, fat, hdl level ...
  + study module relationships
  + find key drivers in interesting modules
+ Limitations
  + assumes linear/monotonic relationships among genes
  + scale-free nw appropriate for protein-coding genes (only)
  + robustness
    + depends on sample size/data quality
