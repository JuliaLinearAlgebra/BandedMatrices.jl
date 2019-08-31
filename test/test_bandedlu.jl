using BandedMatrices, LinearAlgebra, Test, Random
import BandedMatrices: _BandedMatrix

# set prng to some value that avoids test failure
Random.seed!(0)
struct _foo <: Number end


@testset "Banded A\\b" begin
    @testset "banded" begin
        A = brand(5, 1, 1)
        b = Float64[1, 2, 3, 4, 5]

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        @test Af\bf ≈ A\b

        # check A\b has not overwritten b, but has made a copy
        @test b == [1, 2, 3, 4, 5]
    end

    @testset "advanced interface" begin
        # banded
        A = brand(5, 1, 1)
        b = rand(5)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)
        L,U,p = lu(A)
        Lf,Uf,pf = lu(Af)
        @test L ≈ Lf # storage format is different
        @test U ≈ Uf
        @test p ≈ pf
        lua = lu(A)
        @test lu(lua) === lua
        @test Factorization{Float64}(lua) === lua
        @test Matrix(Factorization{Float32}(lua)) ≈ convert(Matrix{Float32}, Af)
        @test Matrix(copy(lua)) ≈ Af
        @test issuccess(lua)

        # note lu makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf ≈ lu(A)\copy(b)
        @test Af\bf ≈ ldiv!(A, copy(b))
        @test Af\bf ≈ ldiv!(lu(A), copy(b))
        @test transpose(Af)\bf ≈ transpose(lu(A))\copy(b)
        @test adjoint(Af)\bf ≈ adjoint(lu(A))\copy(b)
    end

    @testset "conversion of inputs if needed" begin
        # factorisation performs conversion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        @test eltype(lu(Ai)) == Float64

        # no op
        Af = _BandedMatrix(rand(Float32, 3, 5), 5, 1, 1)
        @test eltype(lu(Af)) == Float32

        # linear systems of integer data imply promotion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        bi = collect(1:5)
        @test eltype(Ai\bi) == Float64
        # this code                     ≈ julia base
        @test Ai\bi                     ≈ Matrix(Ai)\copy(bi)
        @test lu(Ai)\bi             ≈ Matrix(Ai)\copy(bi)

        # check A\b makes a copy of b
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        bi = collect(1:5)
        Ai\bi
        @test bi == [1, 2, 3, 4, 5]
    end

    @testset "matrix must be square error" begin
        for (n, m) in zip([7, 5], [5, 7])
            A = brand(m, n, 1, 1)
            b = rand(m)
            @test_throws DimensionMismatch (A\b)
            @test_throws DimensionMismatch lu(A)\b
            @test_throws DimensionMismatch ldiv!(A, b)
            @test_throws DimensionMismatch ldiv!(lu(A), b)
        end
    end

    @testset "transposed algorithm" begin
        # banded
        A = brand(5, 1, 1)
        b = rand(5)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note lu makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af'\bf ≈ ldiv!(transpose(lu(A)),  copy(b))
        @test Af'\bf ≈ ldiv!(lu(A'), copy(b))
    end

    @testset "complex input" begin
        # banded
        A = brand(5, 1, 1) + brand(5, 1, 1)*im
        b = rand(5) + rand(5)*im

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note lu makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ lu(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ lu(A')\copy(b)
        @test Af\bf  ≈ ldiv!(lu(A),  copy(b))
        @test Af'\bf ≈ ldiv!(lu(A'), copy(b))
        @test transpose(Af)\bf ≈ ldiv!(transpose(lu(A)),  copy(b))
        @test adjoint(Af)\bf ≈ ldiv!(lu(A'), copy(b))
        @test adjoint(Af)\bf ≈ ldiv!(lu(A)', copy(b))
    end

    @testset "multiple rhs" begin
        # banded
        A = brand(5, 1, 1)
        b = rand(5, 10)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note lu makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ lu(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ lu(A')\copy(b)
        @test Af\bf  ≈ ldiv!(lu(A),  copy(b))
        @test Af'\bf ≈ ldiv!(lu(A'), copy(b))
        @test Af'\bf ≈ ldiv!(transpose(lu(A)),  copy(b))
        @test Af'\bf ≈ ldiv!(lu(A'), copy(b))
    end

    @testset "properties of factorisation" begin
        BLU = lu(brand(5, 4, 1, 1))
        @test size(BLU) == (5, 4)
        @test size(BLU, 1) == 5
        @test size(BLU, 2) == 4
        @test size(BLU, 3) == 1
        @test_throws BoundsError size(BLU, -1)
        @test_throws BoundsError size(BLU,  0)
    end
end
