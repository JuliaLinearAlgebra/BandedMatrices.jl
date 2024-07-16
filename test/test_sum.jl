module TestSum

using Test, BandedMatrices, Random

Random.seed!(0)
r = brand(rand(1:10_000),rand(1:10_000),rand(-20:100),rand(-20:100))
empty_r = brand(rand(1:1_000),rand(1:1_000),rand(1:100),rand(-200:-101))
n,m = size(empty_r)
matr = Matrix(r)
@testset "sum" begin
    @test sum(empty_r) == 0
    @test sum(empty_r; dims = 2) == zeros(n,1)
    @test sum(empty_r; dims = 1) == zeros(1,m)

    @test sum(r) ≈ sum(matr) rtol = 1e-10
    @test sum(r; dims=2) ≈ sum(matr; dims=2) rtol = 1e-10
    @test sum(r; dims=1) ≈ sum(matr; dims=1) rtol = 1e-10
    @test sum(r; dims=3) == r
    @test_throws ArgumentError sum(r; dims=0)

    v = [1.0]
    sum!(v, r)
    @test v == sum!(v, Matrix(r))
    n2, m2 = size(r)
    v = ones(n2)
    @test sum!(v, r) == sum!(v, Matrix(r))
    V = zeros(1,m2)
    @test sum!(V, r) === V ≈ sum!(zeros(1,m2), Matrix(r))
    V = zeros(n2,m2)
    @test sum!(V, r) === V == r
    @test_throws DimensionMismatch sum!(zeros(Float64, n2 + 1, m2 + 1), r)
end

end