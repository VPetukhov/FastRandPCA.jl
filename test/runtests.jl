using FastRandPCA
using Test
using SparseArrays

@testset "FastRandPCA.jl" begin
    mat = sprand(100, 20, 0.2)

    eigs = eigSVD(mat)
    @assert all(size(eigs.vectors) .== size(mat, 2))
    @assert length(eigs.values) == size(mat, 2)
    @assert all(size(eigs.projection) .== size(mat))

    k = 10
    pcs = pca(mat, k; q=3)
    @assert size(pcs.vectors, 2) == size(mat, 2)
    @assert size(pcs.vectors, 1) == k
    @assert length(pcs.values) == k
    @assert size(pcs.projection, 2) == size(mat, 1)
    @assert size(pcs.projection, 1) == k
    @assert all(size(pcs.projection * mat) .== size(pcs.vectors))
end
