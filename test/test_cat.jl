module TestCat

using BandedMatrices, LinearAlgebra, Test, Random, FillArrays, SparseArrays

@testset "vcat" begin
    @testset "banded matrices" begin
        a = BandedMatrix(0 => 1:2)
        @test vcat(a) == a

        b = BandedMatrix(0 => 1:3,-1 => 1:2, -2 => 1:1)
        @test_throws DimensionMismatch vcat(a,b)

        c = BandedMatrix(0 => [1.0, 2.0, 3.0], 1 => [1.0, 2.0], 2 => [1.0])
        @test eltype(vcat(b, c)) == Float64
        @test vcat(b, c) == vcat(Matrix(b), Matrix(c))

        for i in ((1,2), (-3,4), (0,-1))
            a = BandedMatrix(ones(Float64, rand(1:10), 5), i)
            b = BandedMatrix(ones(Int64, rand(1:10), 5), i)
            c = BandedMatrix(ones(Int32, rand(1:10), 5), i)
            d = vcat(a, b, c)
            sd = vcat(sparse(a), sparse(b), sparse(c))
            @test eltype(d) == Float64
            @test d == sd
            @test bandwidths(d) == bandwidths(sd)
        end
    end

    @testset "one element" begin
        n = rand(3:20)
        x,y = OneElement(1, (1,1), (1,n)), OneElement(1, (1,n), (1,n))
        b = BandedMatrix((0 => ones(n-2), 1 => -2ones(n - 2), 2 => ones(n - 2)), (n-2, n))
        @test vcat(x,b,y) == Tridiagonal([ones(n - 2); 0], [1 ; -2ones(n - 2); 1], [0; ones(n - 2)])
    end
end

end