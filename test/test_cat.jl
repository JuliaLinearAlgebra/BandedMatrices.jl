module TestCat

using BandedMatrices, LinearAlgebra, Test, Random, FillArrays

@testset "vcat" begin
    @testset "banded matrices" begin
        a = BandedMatrix(0 => 1:2)
        @test vcat(a) == a

        b = BandedMatrix(0 => 1:3,-1 => 1:2, -2 => 1:1)
        @test_throws DimensionMismatch vcat(a,b)

        c = BandedMatrix(0 => [1.0, 2.0, 3.0], 1 => [1.0, 2.0], 2 => [1.0])
        @test eltype(vcat(b, c)) == Float64
        @test vcat(b, c) == vcat(Matrix(b), Matrix(c))

        for i = 1:3
            a = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
            b = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
            c = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
            d = vcat(a, b, c)
            @test d == vcat(Matrix(a), Matrix(b), Matrix(c))
            @test bandwidths(d) == (bandwidth(c, 1) + size(a, 1) + size(b, 1), bandwidth(a, 2))
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