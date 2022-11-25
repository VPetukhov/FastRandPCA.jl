using FastRandPCA
using Test
using SparseArrays

@testset "FastRandPCA.jl" begin
    for (n,m) in [[100, 20], [20, 100], [1000, 1000]]
        mat = sprand(n, m, 0.2)

        eigs = eigSVD(mat)
        @test all(size(eigs.V) .== size(mat, 2))
        @test length(eigs.S) == size(mat, 2)
        @test all(size(eigs.U) .== size(mat))

        k = 10
        pcs = pca(mat, k; q=3)
        @test size(pcs.Vt, 2) == size(mat, 2)
        @test size(pcs.Vt, 1) == k
        @test length(pcs.S) == k
        @test size(pcs.U, 2) == size(mat, 1)
        @test size(pcs.U, 1) == k
        @test all(size(pcs.U * mat) .== size(pcs.Vt))
    end
end
