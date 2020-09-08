using BandedMatrices, LinearAlgebra, FillArrays, Test
import BandedMatrices: _BandedMatrix, DefaultBandedMatrix

@testset "misc tests" begin
    @testset "Diagonal of banded" begin
        D = Diagonal([BandedMatrix(Eye(5),(2,3)), BandedMatrix(Eye(6),(1,1))])
        @test @inferred(D[1,1]) == I
        @test @inferred(D[1,2]) == zeros(5,6)
        @test bandwidths(D[1,2]) == (2,1)
    end

    @time @testset "BandedMatrix of BandedMatrix" begin
        A = BandedMatrix{DefaultBandedMatrix{Float64}}(undef, (1, 2), (0, 1))
        A[1,1] = BandedMatrix(Eye(1),(0,1))
        A[1,2] = BandedMatrix(Zeros(1,2),(0,1))
        A[1,2][1,1] = -1/3
        A[1,2][1,2] = 1/3
        B = BandedMatrix{DefaultBandedMatrix{Float64}}(undef, (2, 1), (1, 1))
        B[1,1] = 0.2BandedMatrix(Eye(1),(0,1))
        B[2,1] = BandedMatrix(Zeros(2,1), (1,0))
        B[2,1][1,1] = -2/30
        B[2,1][2,1] = 1/3

        @test (A*B)[1,1][1,1] ≈ 1/3

        A = BandedMatrix(Diagonal([randn(2,3), randn(3,4)]), (0,0))
        @test @inferred(A[1,2]) == zeros(2,4)

        A = BandedMatrix(Diagonal([brand(5,6,2,1), brand(7,8,1,1)]), (0,0))
        @test @inferred(A[1,2]) == zeros(5,8)
    end

    @time @testset "dense overrides" begin
        A = rand(10,11)
        @test bandwidths(A) == (9,10)
        A = rand(10)
        @test bandwidths(A) == (9,0)
        @test bandwidths(A') == (0,9)
    end

    @time @testset "sparse overrides" begin
        for (l,u) = [(0,0), (1,0), (0,1), (3,3), rand(0:10,2)]
            A = brand(10, 10, l, u)
            sA = sparse(A)
            @test sA isa SparseMatrixCSC
            @test bandwidths(sA) == (l,u)
            bA = BandedMatrix(sA)
            @test bA isa BandedMatrix
            @test bA == A
            @test bandwidths(bA) == (l,u)
        end

        for diags = [(-1 => ones(Int, 5),),
                     (-2 => ones(Int, 5),),
                     (2 => ones(Int, 5),),
                     (-1 => ones(Int, 5), 1 => 2ones(Int, 5))]
            A = BandedMatrix(diags...)
            l,u = bandwidths(A)

            sA = sparse(A)
            @test sA isa SparseMatrixCSC
            @test bandwidths(sA) == (l,u)
            bA = BandedMatrix(sA)
            @test bA isa BandedMatrix
            @test bA == A
            @test bandwidths(bA) == (l,u)
        end
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
        @test occursin(Regex("10×10 BandedMatrix{Float64, *$(Matrix{Float64}), *" *
                             string(Base.OneTo{Int})*"}"),
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
