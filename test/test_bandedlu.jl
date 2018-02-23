using BandedMatrices, Compat.Test
import BandedMatrices: _BandedMatrix

# set prng to some value that avoids test failure
srand(0)
struct _foo <: Number end

@testset "Conversion to blas type" begin
    let
        typ = Float64
        @test BandedMatrices._promote_to_blas_type(typ, Complex128) == Complex128
        @test BandedMatrices._promote_to_blas_type(typ, Complex64)  == Complex128
        @test BandedMatrices._promote_to_blas_type(typ, Float64)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Float32)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int64)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int32)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int16)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int8)       == typ

        typ = Float32
        @test BandedMatrices._promote_to_blas_type(typ, Complex128) == Complex128
        @test BandedMatrices._promote_to_blas_type(typ, Complex64)  == Complex64
        @test BandedMatrices._promote_to_blas_type(typ, Float64)    == Float64
        @test BandedMatrices._promote_to_blas_type(typ, Float32)    == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int64)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int32)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int16)      == typ
        @test BandedMatrices._promote_to_blas_type(typ, Int8)       == typ


        @test_throws ErrorException BandedMatrices._promote_to_blas_type(_foo, Float64)
    end

    #
    let
        As   = Any[_BandedMatrix(rand(1:10, 3, 5), 5, 1, 1),
                   _BandedMatrix(rand(3, 5)*im,    5, 1, 1),
                   _BandedMatrix(rand(3, 5),       5, 1, 1)
                  ]
        bs   = Any[rand(1:10,   5),
                   rand(1:10,   5),
                   rand(1:10.0, 5)*im,
                  ]
        typs = Any[Float64,
                   Complex128,
                   Complex128]

        for (A, b, typ) in zip(As, bs, typs)
            AA,   bb   = BandedMatrices._convert_to_blas_type(A,         b)
            AAlu, bblu = BandedMatrices._convert_to_blas_type(lufact(A), b)
            @test eltype(AA) == eltype(bb) == eltype(AAlu) == eltype(bblu) == typ
            @test Matrix(A)\copy(b)             ≈ A\copy(b)
            @test Matrix(A)\copy(b)             ≈ lufact(A)\copy(b)
            @test Matrix(A)\copy(b)             ≈ A_ldiv_B!(A, copy(b))
            @test Matrix(A)\copy(b)             ≈ A_ldiv_B!(lufact(A), copy(b))
            @test transpose(Matrix(A))\copy(b)  ≈ Base.LinAlg.At_ldiv_B!(lufact(A), copy(b))
            @test transpose(Matrix(A))\copy(b)  ≈ Base.LinAlg.A_ldiv_B!(lufact(transpose(A)), copy(b))
            @test adjoint(Matrix(A))\copy(b) ≈ Base.LinAlg.Ac_ldiv_B!(lufact(A), copy(b))
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

        # note lufact makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf ≈ lufact(A)\copy(b)
        @test Af\bf ≈ A_ldiv_B!(A, copy(b))
        @test Af\bf ≈ A_ldiv_B!(lufact(A), copy(b))
    end

    # conversion of inputs if needed
    let
        # factorisation performs conversion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        @test eltype(lufact(Ai)) == Float64

        # no op
        Af = _BandedMatrix(rand(Float32, 3, 5), 5, 1, 1)
        @test eltype(lufact(Af)) == Float32

        # linear systems of integer data imply promotion
        Ai = _BandedMatrix(rand(1:10, 3, 5), 5, 1, 1)
        bi = collect(1:5)
        @test eltype(Ai\bi) == Float64
        # this code                     ≈ julia base
        @test Ai\bi                     ≈ Matrix(Ai)\copy(bi)
        @test lufact(Ai)\bi             ≈ Matrix(Ai)\copy(bi)
        @test A_ldiv_B!(Ai, bi)         ≈ Matrix(Ai)\copy(bi)
        @test A_ldiv_B!(lufact(Ai), bi) ≈ Matrix(Ai)\copy(bi)

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
            @test_throws DimensionMismatch lufact(A)\b
            @test_throws DimensionMismatch A_ldiv_B!(A, b)
            @test_throws DimensionMismatch A_ldiv_B!(lufact(A), b)
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

        # note lufact makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af'\bf ≈ Base.LinAlg.At_ldiv_B!(lufact(A),  copy(b))
        @test Af'\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(A'), copy(b))
    end


    # test complex input algorithm
    let
        # banded
        A = brand(5, 1, 1) + brand(5, 1, 1)*im
        b = rand(5) + rand(5)*im

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note lufact makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ lufact(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ lufact(A')\copy(b)
        @test Af\bf  ≈ Base.LinAlg.A_ldiv_B!(lufact(A),  copy(b))
        @test Af'\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(A'), copy(b))
        @test transpose(Af)\bf ≈ Base.LinAlg.At_ldiv_B!(lufact(A),  copy(b))
        @test adjoint(Af)\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(adjoint(A)), copy(b))
        @test adjoint(Af)\bf ≈ Base.LinAlg.Ac_ldiv_B!(lufact(A), copy(b))
    end

    # test with multiple rhs
    let
        # banded
        A = brand(5, 1, 1)
        b = rand(5, 10)

        # dense storage
        Af = Matrix(A)
        bf = copy(b)

        # note lufact makes copies; these need revision
        # once the lapack storage is built in to a BandedMatrix
        @test Af\bf  ≈ A\copy(b)
        @test Af\bf  ≈ lufact(A)\copy(b)
        @test Af'\bf ≈ A'\copy(b)
        @test Af'\bf ≈ lufact(A')\copy(b)
        @test Af\bf  ≈ Base.LinAlg.A_ldiv_B!(lufact(A),  copy(b))
        @test Af'\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(A'), copy(b))
        @test Af'\bf ≈ Base.LinAlg.At_ldiv_B!(lufact(A),  copy(b))
        @test Af'\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(A'), copy(b))
    end

    # test properties of factorisation
    let
        BLU = lufact(brand(5, 4, 1, 1))
        @test size(BLU) == (5, 4)
        @test size(BLU, 1) == 5
        @test size(BLU, 2) == 4
        @test size(BLU, 3) == 1
        @test_throws ErrorException size(BLU, -1)
        @test_throws ErrorException size(BLU,  0)
    end
end
