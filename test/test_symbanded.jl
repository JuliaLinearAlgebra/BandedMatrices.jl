using BandedMatrices, LinearAlgebra, ArrayLayouts, Random, Test, GenericLinearAlgebra
import BandedMatrices: MemoryLayout, SymmetricLayout, HermitianLayout, BandedColumns

@testset "Symmetric" begin
    A = Symmetric(brand(10,10,1,2))
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  2
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (2,2)
    @test MemoryLayout(typeof(A)) == SymmetricLayout{BandedColumns{DenseColumnMajor}}()

    @test A[1,2] == A[2,1]
    @test A[1,4] == 0
    x=rand(10)
    @test A*x ≈ Matrix(A)*x
    y = similar(x)
    @test all(A*x .=== muladd!(1.0,A,x,0.0,similar(x)) .===
                BLAS.sbmv!('U', 2, 1.0, parent(A).data, x, 0.0, similar(x)))

    @test A[2:10,1:9] isa BandedMatrix
    @test isempty(A[1:0,1:9])
    @test isempty(A[1:9,1:0])
    @test isempty(A[1:0,1:0])

    @test [A[k,j] for k=2:10, j=1:9] == A[2:10,1:9]
    A = Symmetric(brand(10,10,1,2),:L)
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  1
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,1)
    @test MemoryLayout(typeof(A)) == SymmetricLayout{BandedColumns{DenseColumnMajor}}()

    @test A[1,2] == A[2,1]
    @test A[1,3] == 0
    @test A[2:10,1:9] isa BandedMatrix
    @test [A[k,j] for k=2:10, j=1:9] == A[2:10,1:9]

    x=rand(10)
    @test A*x ≈ Matrix(A)*x
    y = similar(x)
    @test all(A*x .=== muladd!(1.0,A,x,0.0,y) .===
                BLAS.sbmv!('L', 1, 1.0, view(parent(A).data,3:4,:), x, 0.0, similar(x)))

    @test norm(A) ≈ norm(Matrix(A))
    @test cond(A) ≈ cond(Matrix(A))

    # (generalized) eigen & eigvals
    Random.seed!(0)

    A = brand(Float64, 100, 100, 2, 4)
    std = eigvals(Symmetric(Matrix(A)))
    @test eigvals(Symmetric(A)) ≈ std
    @test eigvals(Hermitian(A)) ≈ std
    @test eigvals(Hermitian(big.(A))) ≈ std

    A = brand(ComplexF64, 100, 100, 4, 0)
    @test Symmetric(A)[2:10,1:9] isa BandedMatrix
    @test Hermitian(A)[2:10,1:9] isa BandedMatrix
    @test isempty(Hermitian(A)[1:0,1:9])
    @test isempty(Hermitian(A)[1:0,1:0])
    @test isempty(Hermitian(A)[1:9,1:0])
    @test [Symmetric(A)[k,j] for k=2:10, j=1:9] == Symmetric(A)[2:10,1:9]
    @test [Hermitian(A)[k,j] for k=2:10, j=1:9] == Hermitian(A)[2:10,1:9]

    std = eigvals(Hermitian(Matrix(A), :L))
    @test eigvals(Hermitian(A, :L)) ≈ std
    @test eigvals(Hermitian(big.(A), :L)) ≈ std

    A = Symmetric(brand(Float64, 100, 100, 2, 4))
    F = eigen(A)
    Λ, Q = F
    @test Q'Matrix(A)*Q ≈ Diagonal(Λ)
    FD = convert(Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}, F)
    @test FD.vectors'Matrix(A)*FD.vectors ≈ Diagonal(F.values)


    function An(::Type{T}, N::Int) where {T}
        A = Symmetric(BandedMatrix(Zeros{T}(N,N), (0, 2)))
        for n = 0:N-1
            parent(A).data[3,n+1] = T((n+1)*(n+2))
        end
        A
    end

    function Bn(::Type{T}, N::Int) where {T}
        B = Symmetric(BandedMatrix(Zeros{T}(N,N), (0, 2)))
        for n = 0:N-1
            parent(B).data[3,n+1] = T(2*(n+1)*(n+2))/T((2n+1)*(2n+5))
        end
        for n = 0:N-3
            parent(B).data[1,n+3] = -sqrt(T((n+1)*(n+2)*(n+3)*(n+4))/T((2n+3)*(2n+5)*(2n+5)*(2n+7)))
        end
        B
    end

    for T in (Float32, Float64)
        A = An(T, 100)
        B = Bn(T, 100)

        λ = eigvals(A, B)
        @test λ ≈ eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(B)))

        err = λ*(T(2)/π)^2 ./ (1:length(λ)).^2 .- 1

        @test norm(err[1:40]) < 100eps(T)

        Λ, V = eigen(A, B)
        @test V'Matrix(A)*V ≈ Diagonal(Λ)
        @test V'Matrix(B)*V ≈ I
    end

    @testset "eigen with mismatched parent bandwidths" begin
        # A is diagonal
        function eigencheck(SA, SB)
            λS, VS = eigen(SA, SB)
            λ, V = eigen(Matrix(SA), Matrix(SB))
            @test λS ≈ λ ≈ eigvals(SA, SB)
            @test mapreduce((x,y) -> isapprox(abs.(x), abs.(y)), &, eachcol(V), eachcol(VS))
        end
        @testset for (A, uploA) in Any[
                    (BandedMatrix(0=>ones(5), 3=>ones(2)), :L),
                    (BandedMatrix(0=>ones(5), -3=>ones(2)), :U),
                ]
            SA = Symmetric(A, uploA)
            @testset for (B, uploB) in Any[
                        (BandedMatrix(-1=>ones(4), 0=>2ones(5)), :L),
                        (BandedMatrix(0=>2ones(5), 1=>ones(4)), :U),
                    ]
                SB = Symmetric(B, uploB)
                eigencheck(SA, SB)
            end
        end
        # A is non-diagonal. In this case, uplo must match
        @testset for uplo in [:L, :U]
            A = BandedMatrix(-1=>fill(0.1,3), 0=>4ones(5), 1=>fill(0.1,3))
            SA = Symmetric(A, uplo)
            B = BandedMatrix(0=>2ones(5), (uplo == :U ? 1 : -1)=>ones(4))
            SB = Symmetric(B, uplo)
            eigencheck(SA, SB)
        end
    end
end

@testset "Hermitian" begin
    T = ComplexF64
    A = Hermitian(brand(T,10,10,1,2))
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  2
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (2,2)
    @test MemoryLayout(typeof(A)) == HermitianLayout{BandedColumns{DenseColumnMajor}}()

    @test A[1,2] == conj(A[2,1])
    @test A[1,4] == 0
    x=rand(T,10)
    @test A*x ≈ Matrix(A)*x
    @test all(A*x .=== materialize!(MulAdd(one(T),A,x,zero(T),copy(x))) .===
                BLAS.hbmv!('U', 2, T(1.0), parent(A).data, x, T(0.0), similar(x)))

    A = Hermitian(brand(T,10,10,1,2),:L)
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  1
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,1)
    @test MemoryLayout(typeof(A)) == HermitianLayout{BandedColumns{DenseColumnMajor}}()

    @test A[1,2] == conj(A[2,1])
    @test A[1,3] == 0
    x=rand(T, 10)
    @test A*x ≈ Matrix(A)*x

    @test all(A*x .=== muladd!(one(T),A,x,zero(T),copy(x)) .===
                materialize!(MulAdd(one(T),A,x,zero(T),copy(x))) .===
                BLAS.hbmv!('L', 1, one(T), view(parent(A).data,3:4,:), x, zero(T), similar(x)))
end

@testset "LDLᵀ" begin
    for T in (Float16, Float32, Float64, BigFloat, Rational{BigInt})
        A = BandedMatrix{T}(undef,(10,10),(0,2))
        A[band(0)] .= 4
        A[band(1)] .= -one(T)/4
        A[band(2)] .= -one(T)/16
        SAU = Symmetric(A, :U)
        F = factorize(SAU)
        @test F isa LDLt
        b = collect(one(T):size(F, 1))
        x = Matrix(SAU)\b
        y = F\b
        @test x ≈ y
        A = BandedMatrix{T}(undef,(10,10),(2,0))
        A[band(0)] .= 4
        A[band(-1)] .= -one(T)/3
        A[band(-2)] .= -one(T)/9
        SAL = Symmetric(A, :L)
        x = Matrix(SAL)\b
        F = ldlt(SAL)
        y = F\b
        @test x ≈ y
        @test_throws DimensionMismatch F\[b;b]

        T ≠ Float16 && (@test det(F) ≈ det(SAL))
    end
    for T in (Int16, Int32, Int64, BigInt)
        A = BandedMatrix{T}(undef, (4,4), (1,1))
        A[band(0)] .= 3
        A[band(1)] .= 1
        F = ldlt(Symmetric(A))
        @test eltype(F) == float(T)
    end

    @testset "#175" begin
        A = BandedMatrix(0 => [1,2])
        F = ldlt(Symmetric(A))
        @test F.d == [1.,2.]
        @test F.D == Diagonal([1.,2.])
    end
end

@testset "Generalized eigenvalues $W{$T}($Ua,$Ub)($n,$wa-$wb)" for (T,W) in (
                                    (Float32, Symmetric),
                                    (Float64, Symmetric),
                                    #(Float32, Hermitian),
                                    #(Float64, Hermitian),
                                    #(ComplexF32, Hermitian),
                                    #(ComplexF64, Hermitian),
                                   ),
    (Ua, Ub) in  ((:L,:L), (:U,:U)),
    (wa, wb) in ((2, 3), (3, 2)), n in (4,)
    #
    function sbmatrix(W, T, U, w, n)
        r = U == :L ? (0:-1:-w+1) : (0:w-1)
        band(k) = k => ones(T, n - abs(k)) * 2.0^-abs(k)
        W(BandedMatrix(band.(r)...), U)
    end
    A = sbmatrix(W, T, Ua, wa, n)
    B = sbmatrix(W, T, Ub, wb, n)
    @test eigvals(A, B) ≈ eigvals(Matrix(A), Matrix(B))
end

@testset "Cholesky" begin
    for T in (Float64, BigFloat)
        A = Symmetric(BandedMatrix(0 => one(T) ./ [12, 6, 6, 6, 12],
                                1 => ones(T,4) ./ 24))
        Ac = cholesky(A)

        @test Ac isa Cholesky{T,<:BandedMatrix{T}}
        @test Ac.U ≈ cholesky(Matrix(A)).U

        b = rand(T,size(A,1))
        @test Ac\b ≈ Matrix(A)\b
        VERSION >= v"1.9-" && @test Ac\b ≈ A\b
    end

    for T in (Float64, BigFloat)
        A = Symmetric(BandedMatrix(0 => one(T) ./ [12, 6, 6, 6, 12],
                                -1 => ones(T,4) ./ 24), :L)
        Ac = cholesky(A)

        @test Ac isa Cholesky{T,<:BandedMatrix{T}}
        @test Ac.L ≈ cholesky(Matrix(A)).L

        b = rand(T,size(A,1))
        @test Ac\b ≈ Matrix(A)\b
        VERSION >= v"1.9-" && @test Ac\b ≈ A\b
    end

    let T = ComplexF64
        A = Hermitian(BandedMatrix(0 => one(T) ./ [12, 6, 6, 6, 12],
                                   1 => ones(T,4) ./ 24))
        Ac = cholesky(A)

        @test Ac isa Cholesky{T,<:BandedMatrix{T}}
        @test Ac.U ≈ cholesky(Matrix(A)).U

        b = rand(T,size(A,1))
        @test Ac\b ≈ Matrix(A)\b
        VERSION >= v"1.9-" && @test Ac\b ≈ A\b
    end

    @testset "UnitRange" begin
        A = brand(100,100,3,2)
        S = Symmetric(A)
        @test S[3:10,4:11] == Symmetric(Matrix(A))[3:10,4:11]
    end

    @testset "Sym of degenerate bands" begin
        A = SymTridiagonal(zeros(8), fill(0.5,7))
        B = Symmetric(BandedMatrix(1 => fill(0.5,7)))
        @test A ≈ B
        @test BandedMatrices.inbands_getindex(B, 1, 1) == 0
        @test BandedMatrices.inbands_getindex(B, 1, 2) == BandedMatrices.inbands_getindex(B, 2, 1) == 0.5
    end
end
