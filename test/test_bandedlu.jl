using BandedMatrices, Compat.Test, Compat.Random
import BandedMatrices: _BandedMatrix

# set prng to some value that avoids test failure
srand(0)
struct _foo <: Number end

if VERSION < v"0.7-"
    const luf = lufact
    const ldiv! = A_ldiv_B!
    tldiv!(A, b) = At_mul_B!(A, b)
    cldiv!(A, b) = At_mul_B!(A, b)
else
    const luf = lu
    tldiv!(A, b) = ldiv!(transpose(A), b)
    cldiv!(A, b) = ldiv!(adjoint(A), b)
end


@testset "Conversion to blas type" begin
    @testset "_promote_to_blas_type" begin
        typ = Float64
        @test BandedMatrices._promote_to_blas_type(typ, ComplexF64) == ComplexF64
        @test BandedMatrices._promote_to_blas_type(typ, ComplexF32)  == ComplexF64
        @test BandedMatrices._promote_to_blas_type(typ, Float64)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Float32)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int64)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int32)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int16)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int8)       == typ

        typ = Float32
        @test BandedMatrices._promote_to_blas_type(typ, ComplexF64) == ComplexF64
        @test BandedMatrices._promote_to_blas_type(typ, ComplexF32)  == ComplexF32
        @test BandedMatrices._promote_to_blas_type(typ, Float64)    == Float64
        @test BandedMatrices._promote_to_blas_type(typ, Float32)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int64)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int32)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int16)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int8)       == typ


        @test_throws ErrorException BandedMatrices._promote_to_blas_type(_foo, Float64)
    end

    @testset "ldiv!" begin
        As   = Any[_BandedMatrix(rand(1:10, 3, 5), 5, 1, 1),
                   _BandedMatrix(rand(3, 5)*im,    5, 1, 1),
                   _BandedMatrix(rand(3, 5),       5, 1, 1)
                  ]
        bs   = Any[rand(1:10,   5),
                   rand(1:10,   5),
                   rand(1:10.0, 5)*im,
                  ]
        typs = Any[Float64,
                   ComplexF64,
                   ComplexF64]

        for (A, b, typ) in zip(As, bs, typs)
            AA,   bb   = BandedMatrices._convert_to_blas_type(A,         b)
            AAlu, bblu = BandedMatrices._convert_to_blas_type(luf(A), b)
            @test eltype(AA) == eltype(bb) == eltype(AAlu) == eltype(bblu) == typ
            @test Matrix(A)\copy(b)             ≈ A\copy(b)
            @test Matrix(A)\copy(b)             ≈ luf(A)\copy(b)
            @test Matrix(A)\copy(b)             ≈ ldiv!(A, copy(b))
            @test Matrix(A)\copy(b)             ≈ ldiv!(luf(A), copy(b))
            @test transpose(Matrix(A))\copy(b)  ≈ tldiv!(luf(A), copy(b))
            @test transpose(Matrix(A))\copy(b)  ≈ ldiv!(luf(transpose(A)), copy(b))
            @test adjoint(Matrix(A))\copy(b) ≈ cldiv!(luf(A), copy(b))
        end
    end
end

@testset "Banded A\b" begin
    let
        # banded
        A = brand(5, 1, 1)
        b = Float64[1, 2, 3, 4, 5]

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        @test Af\bf ≈ A\b

        # check A\b has not overwritten b, but has made a copy
        @test b == [1, 2, 3, 4, 5]
    end

    # advanced interface
    let
        # banded
        A = brand(5, 1, 1)
        b = rand(5)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note luf makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf ≈ luf(A)\copy(b)
        @test Af\bf ≈ ldiv!(A, copy(b))
        @test Af\bf ≈ ldiv!(luf(A), copy(b))
    end

    # conversion of inputs if needed
    let
        # factorisation performs conversion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        @test eltype(luf(Ai)) == Float64

        # no op
        Af = _BandedMatrix(rand(Float32, 3, 5), 5, 1, 1)
        @test eltype(luf(Af)) == Float32

        # linear systems of integer data imply promotion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        bi = collect(1:5)
        @test eltype(Ai\bi) == Float64
        # this code                     ≈ julia base
        @test Ai\bi                     ≈ Matrix(Ai)\copy(bi)
        @test luf(Ai)\bi             ≈ Matrix(Ai)\copy(bi)
        @test ldiv!(Ai, bi)         ≈ Matrix(Ai)\copy(bi)
        @test ldiv!(luf(Ai), bi) ≈ Matrix(Ai)\copy(bi)

        # check A\b makes a copy of b
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        bi = collect(1:5)
        Ai\bi
        @test bi == [1, 2, 3, 4, 5]
    end

    # matrix must be square error
    let
        for (n, m) in zip([7, 5], [5, 7])
            A = brand(m, n, 1, 1)
            b = rand(m)
            @test_throws DimensionMismatch A\b
            @test_throws DimensionMismatch luf(A)\b
            @test_throws DimensionMismatch ldiv!(A, b)
            @test_throws DimensionMismatch ldiv!(luf(A), b)
        end
    end

    # test transposed algorithm
    let
        # banded
        A = brand(5, 1, 1)
        b = rand(5)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note luf makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af'\bf ≈ tldiv!(luf(A),  copy(b))
        @test Af'\bf ≈ ldiv!(luf(A'), copy(b))
    end


    # test complex input algorithm
    let
        # banded
        A = brand(5, 1, 1) + brand(5, 1, 1)*im
        b = rand(5) + rand(5)*im

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note luf makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ luf(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ luf(A')\copy(b)
        @test Af\bf  ≈ ldiv!(luf(A),  copy(b))
        @test Af'\bf ≈ ldiv!(luf(A'), copy(b))
        @test transpose(Af)\bf ≈ tldiv!(luf(A),  copy(b))
        @test adjoint(Af)\bf ≈ ldiv!(luf(adjoint(A)), copy(b))
        @test adjoint(Af)\bf ≈ cldiv!(luf(A), copy(b))
    end

    # test with multiple rhs
    let
        # banded
        A = brand(5, 1, 1)
        b = rand(5, 10)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note luf makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ luf(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ luf(A')\copy(b)
        @test Af\bf  ≈ ldiv!(luf(A),  copy(b))
        @test Af'\bf ≈ ldiv!(luf(A'), copy(b))
        @test Af'\bf ≈ tldiv!(luf(A),  copy(b))
        @test Af'\bf ≈ ldiv!(luf(A'), copy(b))
    end

    # test properties of factorisation
    let
        BLU = luf(brand(5, 4, 1, 1))
        @test size(BLU) == (5, 4)
        @test size(BLU, 1) == 5
        @test size(BLU, 2) == 4
        @test size(BLU, 3) == 1
        @test_throws ErrorException size(BLU, -1)
        @test_throws ErrorException size(BLU,  0)
    end
end


# if VERSION ≥ v"0.7-"
#     @testset "LU properties" begin
#         A = brand(5,5,4,3); LU = lu(A);
#
#     end
# end
