module FastRandPCA

import IterativeSolvers: lobpcg

using SparseArrays
using LinearAlgebra
using Random

export eigSVD, pca

SVDRes = NamedTuple{(:projection, :values, :vectors), Tuple{Matrix{Float64}, Vector{Float64}, Matrix{Float64}}}

function eigSVD(A::Union{SparseMatrixCSC{T}, Matrix{T}}, k::Int=-1)::SVDRes where T<:Real
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

    return SVDRes((U, d, V))
end

function pca(A::Union{SparseMatrixCSC{T}, Matrix{T}}, k::Int; q::Int=10)::SVDRes where T<:Real
# this is the fast randomized PCA for sparse data
# q is the number of pass over A, q should larger than 1 and q times pass eqauls to (q-2)/2 times power iteration
    q >= 2 || error("Pass parameter q must be larger than 1 !");

    s = 5;
    trans = false
    if size(A, 1) > size(A, 2)
        A = A'
        trans = true
    end

    m,n = size(A);
    if q % 2 == 0
        Q = randn(n, k+s);
        Q = A*Q;
        if q == 2
            Q = eigSVD(Q)[1];
        else
            Q = lu(Q).L;
        end
    else
        Q = randn(m, k+s);
    end
    upper = floor((q-1)/2);
    for i = 1:upper
        if i == upper
            Q = eigSVD(A*(A'*Q))[1];
        else
            Q = lu(A*(A'*Q)).L;
        end
    end
    V,S,U = eigSVD(A'*Q);
    ind = s+1:k+s;
    U = Q * U[:, ind];
    V = V[:, ind];
    S = S[ind];

    if trans
        U,V = V,U
    end

    return SVDRes((U, S, V))
end

end
