module TestSum

using Test, BandedMatrices, Random

r = brand(Float64,rand(1:10_000),rand(1:10_000),rand(-20:100),rand(-20:100))
empty_r = brand(Float64,rand(1:1_000),rand(1:1_000),rand(1:100),rand(-200:-101))
n,m = size(empty_r)
matr = Matrix(r)
@testset "sum" begin
    @test sum(empty_r) == 0
    @test sum(empty_r; dims = 2) == zeros(n,1)
    @test sum(empty_r; dims = 1) == zeros(1,m)
    @test sum(r) == sum(matr)
    @test sum(r; dims=2) == sum(matr; dims=2)
    @test sum(r; dims=1) == sum(matr; dims=1)
    @test sum(r; dims=3) == r
    @test_throws ArgumentError sum(r; dims=0)
end

end