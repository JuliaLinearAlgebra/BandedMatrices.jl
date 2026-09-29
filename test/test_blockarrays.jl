module TestBlockArrays

using BandedMatrices
using ArrayLayouts
using BandedMatrices: _BandedMatrix, BandedColumns
using BlockArrays
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
end

end # module
