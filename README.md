# FastRandPCA.jl

[![Coverage](https://codecov.io/gh/vpetukhov/FastRandPCA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/vpetukhov/FastRandPCA.jl)

Fast randomized PCA algorithms for Sparse Data. It's a julia re-implementation of Matlab [frPCA_sparse](https://github.com/XuFengthucs/frPCA_sparse).

The package includes an implementation of two functions `eig_svd` and `pca`, which are designed to work with sparse matrices (`SparseMatrixCSC`), though can also be applied with normal ones (`Matrix`).

Example code:

```julia
using SparseArrays
eig_svd(sprand(100, 100, 0.2), 10)
```

## Installation

To install it, run `] add FastRandPCA` or `import Pkg; Pkg.add("FastRandPCA")` from Julia REPL. Use `] add FastRandPCA#master` if you want to install a development version.

## References

- Xu Feng, Yuyang Xie, Mingye Song, Wenjian Yu, and Jie Tang, "Fast Randomized PCA for Sparse Data," in Proc. the 10th Asian Conference on Machine Learning (ACML), Beijing, China, Nov. 2018. [10.48550/arXiv.1810.06825](https://doi.org/10.48550/arXiv.1810.06825).
