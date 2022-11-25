module FastRandPCA

import IterativeSolvers: lobpcg

using SparseArrays
using LinearAlgebra
using Random

export eigSVD, pca

"""
Perform truncated singular value decomposition of a matrix `A` to return `k` first values.

Designed to work with sparse matrices. If `k` is not specified, all values are returned.

# Examples
```julia-repl
using SparseArrays
eigSVD(sprand(100, 100, 0.2), 10)
```
"""
function eigSVD(A::AbstractMatrix{<:Number}, k::Int=-1)
    B = A' * A;
    if k == -1
        res = eigen(Matrix(B))
        V, λ = res.vectors, res.values
    else
        r = lobpcg(B, true, k)
        V, λ = r.X, r.λ
    end

    λ[λ .< 0] .= 0 # Some of zero values become negative due to numerical errors
    d::Vector{Float64} = sqrt.(λ);
    U::Matrix{Float64} = ((V' ./ d) * A')';

    return SVD(U, d, V)
end

"""
Perform fast randomized PCA of a matrix `A` to return `k` first components.

Designed to work with sparse matrices.

Parameter `q` is the number of pass over `A`, `q` should larger than 1 and `q` times pass eqauls to `(q-2)/2` times power iteration.

# Examples
```julia-repl
using SparseArrays
pca(sprand(100, 100, 0.2), 10; q=3)
```
"""
function pca(A::AbstractMatrix{<:Number}, k::Int; q::Int=10)
    q >= 2 || error("Pass parameter q must be larger than 1 !");

    s = 5;
    if size(A, 1) > size(A, 2)
        r = pca(A', k; q=q)
        return SVD(Matrix(r.Vt), r.S, Matrix(r.U))
    end

    m,n = size(A);
    if q % 2 == 0
        Q = randn(n, k+s);
        Q = A*Q;
        if q == 2
            Q = eigSVD(Q).U;
        else
            Q = lu(Q).L;
        end
    else
        Q = randn(m, k+s);
    end
    upper = floor((q-1)/2);
    for i = 1:upper
        if i == upper
            Q = eigSVD(A*(A'*Q)).U;
        else
            Q = lu(A*(A'*Q)).L;
        end
    end
    V,S,U = eigSVD(A'*Q);
    ind = s+1:k+s;
    U = (Q * U[:, ind])';
    V = V[:, ind]';
    S = S[ind];

    return SVD(U, S, V)
end

end
