using BandedMatrices, LinearAlgebra, FillArrays, Test
import BandedMatrices: _BandedMatrix
## Banded Matrix of Banded Matrix

BandedMatrixWithZero = Union{BandedMatrix{Float64,Matrix{Float64}}, UniformScaling}
# need to define the concept of zero
Base.zero(::Type{BandedMatrixWithZero}) = 0*I

@testset "misc tests" begin
    @time @testset "BandedMatrixWithZero" begin
        A = BandedMatrix{BandedMatrixWithZero}(undef, (1, 2), (0, 1))
        A[1,1] = BandedMatrix(Eye(1),(0,1))
        A[1,2] = BandedMatrix(Zeros(1,2),(0,1))
        A[1,2][1,1] = -1/3
        A[1,2][1,2] = 1/3
        B = BandedMatrix{BandedMatrixWithZero}(undef, (2, 1), (1, 1))
        B[1,1] = 0.2BandedMatrix(Eye(1),(0,1))
        B[2,1] = BandedMatrix(Zeros(2,1), (1,0))
        B[2,1][1,1] = -2/30
        B[2,1][2,1] = 1/3

        # A*B has insane compile time in v0.7-alpha
        @test_skip (A*B)[1,1][1,1] ≈ 1/3
    end

    @time @testset "dense overrides" begin
        A = rand(10,11)
        @test bandwidths(A) == (9,10)
        A = rand(10)
        @test bandwidths(A) == (9,0)
        @test bandwidths(A') == (0,9)
    end

    @time @testset "trivial convert routines" begin
        A = brand(3,4,1,2)
        @test isa(BandedMatrix{Float64}(A), BandedMatrix{Float64})
        @test isa(AbstractMatrix{Float64}(A), BandedMatrix{Float64})
        @test isa(AbstractArray{Float64}(A), BandedMatrix{Float64})
        @test isa(BandedMatrix(A), BandedMatrix{Float64})
        @test isa(AbstractMatrix(A), BandedMatrix{Float64})
        @test isa(AbstractArray(A), BandedMatrix{Float64})
        @test isa(BandedMatrix{ComplexF16}(A), BandedMatrix{ComplexF16})
        @test isa(AbstractMatrix{ComplexF16}(A), BandedMatrix{ComplexF16})
        @test isa(AbstractArray{ComplexF16}(A), BandedMatrix{ComplexF16})
    end
    @time @testset "show" begin
      @test occursin("10×10 BandedMatrix{Float64,Array{Float64,2},"*string(Base.OneTo{Int})*"}",
         sprint() do io
            show(io, MIME"text/plain"(), brand(10, 10, 3, 3))
         end)
        needle = "1.0  0.0   ⋅ \n 0.0  1.0  0.0\n  ⋅   0.0  1.0"
        @test occursin(needle, sprint() do io
             show(io, MIME"text/plain"(), BandedMatrix(Eye(3),(1,1)))
          end)
    end
    @time @testset "Issue #27" begin
        A=brand(1,10,0,9)
        B=brand(10,10,255,255)
        @test Matrix(A*B)  ≈ Matrix(A)*Matrix(B)
    end

    @testset "defaultdot" begin
        A = randn(5)
        @test BandedMatrices.dot(A,A) ≡ LinearAlgebra.dot(A,A)
    end

    @testset "Offset axes" begin
        A = _BandedMatrix(Ones((Base.OneTo(3),Base.Slice(-5:5),)), Base.Slice(-4:2), 1,1)
        @test size(A) == (7,11)
        @test A[-4,-5] == 1
        @test A[-4,-5+2] == 1
        @test A[-4,-5+3] == 0
        @test_throws BoundsError A[-5,-5+3]
    end
end
