using FastRandPCA
using Test
using SparseArrays

@testset "FastRandPCA.jl" begin
    for (n,m) in [[100, 20], [20, 100], [1000, 1000]]
        mat = sprand(n, m, 0.2)

        eigs = eig_svd(mat)
        @test all(size(eigs.V) .== size(mat, 2))
        @test length(eigs.S) == size(mat, 2)
        @test all(size(eigs.U) .== size(mat))

        k = 10
        for exact in [false, true]
            pcs = pca(mat, k; q=3, exact_svd=exact)
            @test size(pcs.Vt, 2) == size(mat, 2)
            @test size(pcs.Vt, 1) == k
            @test length(pcs.S) == k
            @test size(pcs.U, 2) == size(mat, 1)
            @test size(pcs.U, 1) == k
            @test all(size(pcs.U * mat) .== size(pcs.Vt))
        end

        for MT in [SparseMatrixCSC{Float32, Int32}, Matrix{Float32}]
            pcs = pca(MT(mat), k; q=3)
            @test eltype(pcs.U) == Float32
            @test eltype(pcs.V) == Float32
            @test eltype(pcs.S) == Float32
        end
    end
end
