using Test, BandedMatrices, Random

r = brand(Float64,rand(1:10_000),rand(1:10_000),rand(-20:100),rand(-20:100))
matr = Matrix(r)
@testset "sum" begin
    @test sum(r) ≈ sum(matr) atol = 1e-10
    @test sum(r; dims=2) ≈ sum(matr; dims=2) atol = 1e-10
    @test sum(r; dims=1) ≈ sum(matr; dims=1) atol = 1e-10
    @test sum(r; dims=3) == r
    @test_throws ArgumentError sum(r; dims=0)
end