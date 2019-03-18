using BandedMatrices, LinearAlgebra, LazyArrays, Random, Test
    import LazyArrays: DenseColumnMajor
    import BandedMatrices: MemoryLayout, SymmetricLayout, HermitianLayout, BandedColumns


@testset "Symmetric" begin
    A = Symmetric(brand(10,10,1,2))
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  2
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (2,2)
    @test MemoryLayout(A) == SymmetricLayout(BandedColumns(DenseColumnMajor()), 'U')

    @test A[1,2] == A[2,1]
    @test A[1,4] == 0
    x=rand(10)
    @test A*x ≈ Matrix(A)*x
    @test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= 1.0.*Mul(A,x) .+ 0.0.*similar(x)) .===
                BLAS.sbmv!('U', 2, 1.0, parent(A).data, x, 0.0, similar(x)))

    A = Symmetric(brand(10,10,1,2),:L)
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  1
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,1)
    @test MemoryLayout(A) == SymmetricLayout(BandedColumns(DenseColumnMajor()), 'L')

    @test A[1,2] == A[2,1]
    @test A[1,3] == 0
    x=rand(10)
    @test A*x ≈ Matrix(A)*x

    @test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= 1.0.*Mul(A,x) .+ 0.0.*similar(x)) .===
                BLAS.sbmv!('L', 1, 1.0, view(parent(A).data,3:4,:), x, 0.0, similar(x)))

    @test norm(A) ≈ norm(Matrix(A))
    @test cond(A) ≈ cond(Matrix(A))

    # (generalized) eigen & eigvals
    Random.seed!(0)

    A = Symmetric(brand(Float64, 10, 10, 0, 4))
    @test eigvals(A) ≈ eigvals(Symmetric(Matrix(A)))

    Λ, Q = eigen(A)
    @test Q'A*Q ≈ Diagonal(Λ)

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
        @test V'A*V ≈ Diagonal(Λ)
        @test V'B*V ≈ I
    end
end



@testset "Hermitian" begin
    T = ComplexF64
    A = Hermitian(brand(T,10,10,1,2))
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  2
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (2,2)
    @test MemoryLayout(A) == HermitianLayout(BandedColumns(DenseColumnMajor()), 'U')

    @test A[1,2] == conj(A[2,1])
    @test A[1,4] == 0
    x=rand(T,10)
    @test A*x ≈ Matrix(A)*x
    @test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= T(1.0).*Mul(A,x) .+ T(0.0).*similar(x)) .===
                BLAS.hbmv!('U', 2, T(1.0), parent(A).data, x, T(0.0), similar(x)))

    A = Hermitian(brand(T,10,10,1,2),:L)
    @test isbanded(A)
    @test BandedMatrix(A) == A
    @test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  1
    @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,1)
    @test MemoryLayout(A) == HermitianLayout(BandedColumns(DenseColumnMajor()), 'L')

    @test A[1,2] == conj(A[2,1])
    @test A[1,3] == 0
    x=rand(T, 10)
    @test A*x ≈ Matrix(A)*x

    @test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= one(T).*Mul(A,x) .+ zero(T).*similar(x)) .===
                BLAS.hbmv!('L', 1, one(T), view(parent(A).data,3:4,:), x, zero(T), similar(x)))
end

@testset "LDLᵀ" begin
    for T in (Float16, Float32, Float64, BigFloat, Rational{BigInt})
        A = BandedMatrix{T}(undef,(10,10),(0,2))
        A[band(0)] .= 4
        A[band(1)] .= -one(T)/4
        A[band(2)] .= -one(T)/16
        SAU = Symmetric(A, :U)
        F = ldlt(SAU)
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
        @test det(F) ≈ det(SAL)
    end
    for T in (Int16, Int32, Int64, BigInt)
        A = BandedMatrix{T}(undef, (4,4), (1,1))
        A[band(0)] .= 3
        A[band(1)] .= 1
        F = ldlt(Symmetric(A))
        @test eltype(F) == float(T)
    end
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
        @test_broken Ac\b ≈ A\b
    end 
    
    for T in (Float64, BigFloat) 
        A = Symmetric(BandedMatrix(0 => one(T) ./ [12, 6, 6, 6, 12],
                                -1 => ones(T,4) ./ 24), :L)
        Ac = cholesky(A)

        @test Ac isa Cholesky{T,<:BandedMatrix{T}}                           
        @test Ac.L ≈ cholesky(Matrix(A)).L

        b = rand(T,size(A,1))
        @test Ac\b ≈ Matrix(A)\b
        @test_broken Ac\b ≈ A\b
    end 
end
