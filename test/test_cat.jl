module TestCat

using BandedMatrices, LinearAlgebra, Test, Random

@testset "vcat" begin
    a = BandedMatrix(0 => 1:2)
    @test vcat(a) == a

    b = BandedMatrix(0 => 1:3,-1 => 1:2, -2 => 1:1)
    @test_throws DimensionMismatch vcat(a,b)

    c = BandedMatrix(0 => [1.0, 2.0, 3.0], 1 => [1.0, 2.0], 2 => [1.0])
    @test eltype(vcat(b,c)) == Float64
    @test vcat(b,c) == vcat(Matrix(b),Matrix(c))

    for i = 1:3
        a = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
        b = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
        c = brand(Float64, rand(1:10), 5, rand(1:10),rand(-4:4))
        @test vcat(a,b,c) == vcat(Matrix(a),Matrix(b),Matrix(c))
    end
end

end