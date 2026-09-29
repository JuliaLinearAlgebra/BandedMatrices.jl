module TestBlockArrays

using BandedMatrices
using ArrayLayouts
using BandedMatrices: _BandedMatrix, BandedColumns
using BlockArrays
using InfiniteArrays
using LinearAlgebra
using Test

@testset "BandedMatrix with blocked axes" begin
    ax = blockedrange(1:4)
    @testset "undef" begin
        A = BandedMatrix{Float64}(undef, (ax, ax), (1,1))
        @test A.data isa BlockedMatrix
        @test blockisequal(axes(A), (ax, ax))
        @test bandwidths(A) == (1,1)
        @test MemoryLayout(A) isa BandedColumns{DenseColumnMajor}

        B = BandedMatrix{Float64}(undef, (Base.OneTo(3), ax), (1,1))
        @test axes(B,1) ≡ Base.OneTo(3)
        @test blockisequal(axes(B,2), ax)

        @test BandedMatrix{BigFloat}(undef, (ax, ax), (1,1)) == zeros(10,10)

        V = BandedMatrix{Vector{Float64}}(undef, (ax, ax), (1,1))
        @test V.data isa BlockedMatrix{Vector{Float64}}
        @test blockisequal(axes(V), (ax, ax))
    end

    @testset "multiplication" begin
        A = BandedMatrix{Float64}(undef, (ax, ax), (1,1))
        A.data .= randn.()
        @test A[Block(2,2)] == Matrix(A)[2:3,2:3]
        AA = A*A
        @test AA isa BandedMatrix
        @test blockisequal(axes(AA), (ax, ax))
        @test bandwidths(AA) == (2,2)
        @test AA ≈ Matrix(A)^2
    end

    @testset "similar" begin
        A = BandedMatrix{Float64}(undef, (ax, ax), (1,1))
        for M in (MulAdd(A, A), MulAdd(UpperTriangular(A), A), MulAdd(Symmetric(A), A))
            S = similar(M, Float64, (ax, ax))
            @test S isa BandedMatrix
            @test blockisequal(axes(S), (ax, ax))
        end
        D = Diagonal(randn(10))
        @test blockisequal(axes(similar(MulAdd(D, D), Float64, (ax, ax))), (ax, ax))
        # infinite axes use the default
        B = brand(5,5,1,1)
        S = similar(MulAdd(B, B), Float64, (InfiniteArrays.OneToInf(), InfiniteArrays.OneToInf()))
        @test !(S isa BandedMatrix)
        @test size(S) == (ℵ₀, ℵ₀)
    end
end

end # module
