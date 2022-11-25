# FastRandPCA.jl

[![Coverage](https://codecov.io/gh/vpetukhov/FastRandPCA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vpetukhov/FastRandPCA.jl)

Fast randomized PCA algorithms for Sparse Data. It's a julia re-implementation of Matlab [frPCA_sparse](https://github.com/XuFengthucs/frPCA_sparse).

The package includes an implementation of two functions `eigSVD` and `pca`, which are designed to work with sparse matrices (`SparseMatrixCSC`), though can also be applied with normal ones (`Matrix`).

## References

- Xu Feng, Yuyang Xie, Mingye Song, Wenjian Yu, and Jie Tang, "Fast Randomized PCA for Sparse Data," in Proc. the 10th Asian Conference on Machine Learning (ACML), Beijing, China, Nov. 2018. [10.48550/arXiv.1810.06825](https://doi.org/10.48550/arXiv.1810.06825).