module FastRandPCA

using KrylovKit: svdsolve, KrylovDefaults

using SparseArrays
using LinearAlgebra
using Random

export eig_svd, pca

"""
`eig_svd(A, [k]; [kwargs...])`

Perform singular value decomposition of a matrix `A` by estimating `svd(A' * A)`. If `k` is provided,
Krylov iterative solver is used to return only `k` largest singular values and corresponding singular vectors.

Designed to work with sparse matrices. Keyword arguments are forwarded to `KrylovKit.svdsolve`.

# Examples

```julia-repl
using SparseArrays
eig_svd(sprand(100, 100, 0.2), 10)
```
"""
function eig_svd(A::AbstractMatrix{<:Number}, k::Int=-1; krylovdim::Int=-1, kwargs...)
    B = A' * A;
    if (k == -1) || (k == size(B, 1))
        res = eigen(Matrix(B))
        V, λ = res.vectors, res.values
    else
        if krylovdim <= 0
            krylovdim = max(KrylovDefaults.krylovdim, k + 2)
        end
        r = svdsolve(B, size(B, 1), k, krylovdim=krylovdim, kwargs...)
        V, λ = hcat(r[2]...), r[1]
    end

    λ[λ .< 0] .= 0 # Some of zero values become negative due to numerical errors
    d = sqrt.(λ);
    U::Matrix = ((V' ./ d) * A')';

    return SVD(U, d, V)
end

svd_wrap(A::AbstractMatrix{<:Number}, exact_svd::Bool) = (exact_svd ? svd(A) : eig_svd(A))

"""
`pca(A, k; [q=10], [exact_svd=false], [s=5])`

Perform fast randomized PCA of a matrix `A` to return `k` first components.

Designed to work with sparse matrices.

# Keyword arguments

- `exact_svd` - if `true`, use exact SVD on random projections, otherwise use `eig_svd` approximation.
  Setting it to `true` would improve precision of the algorithm on small eigenvalues, but would drastically decrease performance
  on large matrices (>1M rows).
- `q` is the number of pass over `A`, `q` should larger than 1 and `q` times pass eqauls to `(q-2)/2` times power iteration
- `s` is the excess dimension for the random matrix. This parameter is not supposed to be changed normally.

# Examples

```julia-repl
using SparseArrays
pca(sprand(100, 100, 0.2), 10; q=3)
```
"""
function pca(A::AbstractMatrix{<:Number}, k::Int; q::Int=10, exact_svd::Bool=false, s=5)
    q >= 2 || error("Pass parameter q must be larger than 1 !");

    if size(A, 1) > size(A, 2)
        r = pca(A', k; q=q)
        return SVD(Matrix(r.Vt), r.S, Matrix(r.U))
    end

    m,n = size(A);
    if q % 2 == 0
        Q = randn(promote_type(Float32, eltype(A)), n, k+s);
        Q = A*Q;
        if q == 2
            Q = svd_wrap(Q, exact_svd).U;
        else
            Q = lu(Q).L;
        end
    else
        Q = randn(promote_type(Float32, eltype(A)), m, k+s);
    end
    upper = floor((q-1)/2);
    for i = 1:upper
        if i == upper
            Q = svd_wrap(A*(A'*Q), exact_svd).U;
        else
            Q = lu(A*(A'*Q)).L;
        end
    end
    V,S,U = svd_wrap(A'*Q, exact_svd);
    ind = s+1:k+s;
    U = (Q * U[:, ind])';
    V = V[:, ind]';
    S = S[ind];

    return SVD(U, S, V)
end

end
