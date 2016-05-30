using BandedMatrices, Base.Test

# conversion to blas type
type _foo <: Number end
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

# conversion of inputs to appropriate blas type
let
    As   = Any[BandedMatrix([7 1 10 5 3;2 9 4 5 4;6 6 4 8 3],5, 1, 1),
               BandedMatrix(Complex{Float64}[0.0 + 0.31031554631348945im 0.0 + 0.08825549765396867im 0.0 + 0.45053973810477266im 0.0 + 0.5839629261767354im 0.0 + 0.14930808711787047im
                 0.0 + 0.035019092209872094im 0.0 + 0.7831589671127186im 0.0 + 0.5511193450358627im 0.0 + 0.10000265367777472im 0.0 + 0.34682402108740806im
                 0.0 + 0.6489233684971665im 0.0 + 0.8912848313765389im 0.0 + 0.09099934491944817im 0.0 + 0.5588005479153957im 0.0 + 0.801264900502275im],    5, 1, 1),
               BandedMatrix([0.8904594888624968 0.3856257470414042 0.8858218678249399 0.680310946023321 0.11767396680452924
 0.05322615728943747 0.5703362475732607 0.7360106041746182 0.11768822472293161 0.01965702815691306
 0.8793158621464643 0.26054678719557467 0.46205900187581506 0.9226184343172599 0.9305149562765167],       5, 1, 1)
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
        @test full(A)\copy(b)             ≈ A\copy(b)
        @test full(A)\copy(b)             ≈ lufact(A)\copy(b)
        @test full(A)\copy(b)             ≈ A_ldiv_B!(A, copy(b))
        @test full(A)\copy(b)             ≈ A_ldiv_B!(lufact(A), copy(b))
        @test transpose(full(A))\copy(b)  ≈ Base.LinAlg.At_ldiv_B!(lufact(A), copy(b))
        @test transpose(full(A))\copy(b)  ≈ Base.LinAlg.A_ldiv_B!(lufact(transpose(A)), copy(b))
        @test ctranspose(full(A))\copy(b) ≈ Base.LinAlg.Ac_ldiv_B!(lufact(A), copy(b))
    end
end

# basic A\b interface
let
    # banded
    A = BandedMatrix([0.9109574042615387 0.2531673195776303 0.6070200736073808 0.9683264981952036 0.5370310690623921
 0.09517935950517886 0.917567430064258 0.468859960625579 0.7083524502288101 0.053723677324431174
 0.8081326015121129 0.9786127501091892 0.1292803635413109 0.7366627713526539 0.938873900305708],
        5,1,1)

    b = Float64[1, 2, 3, 4, 5]

    # dense storage
    Af = full(A)
    bf = copy(b)

    @test Af\bf ≈ A\b

    # check A\b has not overwritten b, but has made a copy
    @test b == [1, 2, 3, 4, 5]
end

# advanced interface
let
    # banded
    A = BandedMatrix([0.9612554787929786 0.33278183861159416 0.2874123608196659 0.4656930776312018 0.019948804577760715
 0.21742253390807975 0.026233794511437702 0.956939241014354 0.21745355177760906 0.7269728248012044
 0.24069521506774771 0.8898298753194889 0.9903986326080136 0.18005941867445174 0.10911235710030942],5, 1, 1)

    b = rand(5)

    # dense storage
    Af = full(A)
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
    Ai = BandedMatrix([3 2 8 5 4
                        10 4 3 6 2
                        7 8 7 1 6], 5, 1, 1)
    @test eltype(lufact(Ai)) == Float64

    # no op
    Af = BandedMatrix(Float32[0.6347754 0.14602828 0.7878281 0.41491115 0.48483682
        0.67632616 0.013396502 0.5766399 0.8053093 0.083145976
        0.8108088 0.7455702 0.60512173 0.52822673 0.06679964], 5, 1, 1)
    @test eltype(lufact(Af)) == Float32

    # linear systems of integer data imply promotion
    Ai = BandedMatrix([8 7 9 4 6
 2 3 4 5 5
 7 2 2 1 7], 5, 1, 1)

    bi = collect(1:5)
    @test eltype(Ai\bi) == Float64
    # this code                     ≈ julia base
    @test Ai\bi                     ≈ full(Ai)\copy(bi)
    @test lufact(Ai)\bi             ≈ full(Ai)\copy(bi)
    @test A_ldiv_B!(Ai, bi)         ≈ full(Ai)\copy(bi)
    @test A_ldiv_B!(lufact(Ai), bi) ≈ full(Ai)\copy(bi)

    # check A\b makes a copy of b
    Ai = BandedMatrix([10 2 1 5 2
 2 2 5 10 6
 4 7 4 9 10], 5, 1, 1)

    bi = collect(1:5)
    Ai\bi
    @test bi == [1, 2, 3, 4, 5]
end

# matrix must be square error
let
    for (n, m) in zip([7, 5], [5, 7])
        A = brand(m, n, 1, 1)
        b = rand(m)
        @test_throws ArgumentError A\b
        @test_throws ArgumentError lufact(A)\b
        @test_throws ArgumentError A_ldiv_B!(A, b)
        @test_throws ArgumentError A_ldiv_B!(lufact(A), b)
    end
end

# test transposed algorithm
let
    # banded
    A = BandedMatrix([0.9263024383608169 0.9033628694250249 0.5850597226672523 0.5489829246108773 0.8818306693459987
 0.5493194170533449 0.8637677543532063 0.34489462768234525 0.7623021395064611 0.136428752941945
 0.31460022859311043 0.7941778206157795 0.7590649827240794 0.06807703849722913 0.18820897241754686],5, 1, 1)

    b = rand(5)

    # dense storage
    Af = full(A)
    bf = copy(b)

    # note lufact makes copies; these need revision
    # once the lapack storage is built in to a BandedMatrix
    @test Af'\bf ≈ Base.LinAlg.At_ldiv_B!(lufact(A),  copy(b))
    @test Af'\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(A'), copy(b))
end


# test complex input algorithm
let
    # banded
    A = BandedMatrix(Complex{Float64}[0.0 + 0.0im 0.16640874328332167 + 0.2724862391590359im 0.79332713506196 + 0.017360719450438378im 0.023232457920397964 + 0.3519624887498083im 0.8431337556047906 + 0.7418612583732505im
                 0.12682328906620732 + 0.9030127433689894im 0.4327350249018864 + 0.1945500964130611im 0.10333593914887018 + 0.059082272883393516im 0.5653983022631415 + 0.30918847438993646im 0.3292070514888883 + 0.585498893024105im
                 0.914099614371781 + 0.30934696917524485im 0.29674562903852775 + 0.30569169837779153im 0.4962631670699633 + 0.6495278791421764im 0.7331276404670288 + 0.8089571342994171im 0.0 + 0.0im],5, 1, 1)
    b = rand(5) + rand(5)*im


    # dense storage
    Af = full(A)
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
    @test ctranspose(Af)\bf ≈ Base.LinAlg.A_ldiv_B!(lufact(ctranspose(A)), copy(b))
    @test ctranspose(Af)\bf ≈ Base.LinAlg.Ac_ldiv_B!(lufact(A), copy(b))
end

# test with multiple rhs
let
    # banded
    A = BandedMatrix([0.3660440932692133 0.505433255478682 0.09164176986860628 0.5755999588881693 0.9734336278316384
 0.7858032445551113 0.9213064973875218 0.7730153879550115 0.6890468957597959 0.7380274244817611
 0.03631986175228996 0.6341314732629972 0.06114923945360373 0.3256294517935441 0.8385617260595446],5, 1, 1)
    b = rand(5, 10)

    # dense storage
    Af = full(A)
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
    A = BandedMatrix([0.4224089399360216 0.9005758424800507 0.170365675702024 0.98381909870464
 0.19314062696832424 0.7361152224231435 0.0946532708225285 0.8733519312891731
 0.7250301000415993 0.9522891774821141 0.7153012598425206 0.5396723761182507],5, 1, 1)

    BLU = lufact(A)


    @test size(BLU) == (5, 4)
    @test size(BLU, 1) == 5
    @test size(BLU, 2) == 4
    @test size(BLU, 3) == 1
    @test_throws ErrorException size(BLU, -1)
    @test_throws ErrorException size(BLU,  0)
end
