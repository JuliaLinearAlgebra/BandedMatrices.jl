using BandedMatrices, LinearAlgebra, LazyArrays, Test
    import LazyArrays: MemoryLayout, MulAdd, DenseColumnMajor, ConjLayout
    import Base: BroadcastStyle
    import BandedMatrices: BandedStyle, BandedRows

@testset "broadcasting" begin
    @testset "general" begin
        n = 1000
        A = brand(n,n,1,1)
        B = Matrix{Float64}(undef, n,n)
        B .= exp.(A)
        @test B == exp.(Matrix(A)) == exp.(A)

        @test exp.(A) isa BandedMatrix
        @test bandwidths(exp.(A)) == (n-1,n-1)

        C = similar(A)
        @test_throws BandError C .= exp.(A)

        @test identity.(A) isa BandedMatrix
        @test bandwidths(identity.(A)) == bandwidths(A)

        @test (z -> exp(z)-1).(A) isa BandedMatrix
        @test bandwidths((z -> exp(z)-1).(A)) == bandwidths(A)

        @test A .+ 1 isa BandedMatrix
        @test (A .+ 1) == Matrix(A) .+ 1

        @test A ./ 1 isa BandedMatrix
        @test bandwidths(A ./ 1) == bandwidths(A)

        @test 1 .+ A isa BandedMatrix
        @test (1 .+ A) == 1 .+ Matrix(A)

        @test 1 .\ A isa BandedMatrix
        @test bandwidths(1 .\ A) == bandwidths(A)

        A = brand(10,1,1,1)
        @test A[:,1] isa Vector
        @test norm(A .- A[:,1]) == 0
        @test A ≈ A[:,1]
    end

    @testset "identity" begin
        n = 100
        A = brand(n,n,2,2)
        A.data[1,1] = NaN
        B = brand(n,n,2,2)
        B.data[1,1] = NaN
        A .= B
        @test A == B
        B = brand(n,n,1,1)
        B.data[1,1] = NaN
        A .= B
        B = brand(n,n,3,3)
        B[band(3)] .= 0
        B[band(-3)] .= 0
        B.data[1,1] = NaN
        B.data[end,end] = NaN
        A .= B
        @test A == B

        B = brand(n,n,0,3)
        B[band(3)] .= 0
        B.data[1,1] = NaN
        A .= B
        @test A == B

        B = brand(n,n,3,0)
        B[band(-3)] .= 0
        B.data[end,end] = NaN
        A .= B
        @test A == B

        B = brand(n,n,-1,1)
        A .= B
        @test A == B

        B = brand(n,n,1,-1)
        A .= B
        @test A == B
    end

    @testset "lmul!/rmul!" begin
        n = 1000
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B .= (-).(A)
        @test -A isa BandedMatrix
        @test (-).(A) isa BandedMatrix
        @test bandwidths(A) == bandwidths(-A) == bandwidths((-).(A))
        @test B == -A == (-).(A)
        @test A-I isa BandedMatrix
        @test I-A isa BandedMatrix
        @test bandwidths(A) == bandwidths(A-I) == bandwidths(I-A)

        n = 1000
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B .= 2.0.*A

        @test B ==  2A == 2.0.*A
        @test 2A isa BandedMatrix
        @test 2.0.*A isa BandedMatrix
        @test bandwidths(2A) == bandwidths(2.0.*A) == bandwidths(A)

        A .= 2.0.*A
        @test A == B

        n = 1000
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B .= A.*2.0

        @test B ==  A*2 == A.*2.0
        @test A*2 isa BandedMatrix
        @test A .* 2.0 isa BandedMatrix
        @test bandwidths(A*2) == bandwidths(A.*2.0) == bandwidths(A)
        A .= A.*2.0
        @test A == B

        n = 1000
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B .= A ./ 2.0

        @test B == A/2 == A ./ 2.0
        @test A/2 isa BandedMatrix
        @test A ./ 2.0 isa BandedMatrix
        @test bandwidths(A/2) == bandwidths(A ./ 2.0) == bandwidths(A)

        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B .= 2.0 .\ A

        @test B == A/2 == A ./ 2.0
        @test 2\A isa BandedMatrix
        @test 2.0 .\ A isa BandedMatrix
        @test bandwidths(2\A) == bandwidths(2.0 .\ A) == bandwidths(A)

        A = brand(5,5,1,1)
        A.data .= NaN
        lmul!(0.0,A)
        @test isnan(norm(A)) == isnan(norm(lmul!(0.0,[NaN])))

        lmul!(false,A)
        @test norm(A) == 0.0

        A = brand(5,5,1,1)
        A.data .= NaN
        rmul!(A,0.0)
        @test isnan(norm(A)) == isnan(norm(rmul!([NaN],0.0)))

        rmul!(A,false)
        @test norm(A) == 0.0

        n = 100
        A = brand(n,n,2,2)
        B = brand(n,n,1,1)
        A[band(2)] .= A[band(-2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)

        n = 100
        A = brand(n,n,2,2)
        B = brand(n,n,1,3)
        A[band(-2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)

        A = brand(n,n,2,2)
        B = brand(n,n,3,1)
        A[band(2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)
    end

    @testset "axpy!" begin
        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        C .= A .+ B
        @test C == A + B == A .+ B  == Matrix(A) + Matrix(B)
        @test A + B isa BandedMatrix
        @test A .+ B isa BandedMatrix
        @test bandwidths(A+B) == bandwidths(A.+B) == (2,2)
        B .= A .+ B
        @test B == C

        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        C .= A .* B
        @test C == A .* B  == Matrix(A) .* Matrix(B)
        @test A .* B isa BandedMatrix
        @test bandwidths(A.*B) == (2,2)
        @time B .= A .* B
        @test B == C

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(2)] .= B[band(-2)] .= 0
        C = brand(n,n,1,1)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(-2)] .= 0
        C = brand(n,n,1,2)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(2)] .= 0
        C = brand(n,n,2,1)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)

        C .= 2.0 .* A .+ B
        @test C == 2A+B == 2.0.*A .+ B

        @test 2A + B isa BandedMatrix
        @test 2.0.*A .+ B isa BandedMatrix
        @test bandwidths(2A+B) == bandwidths(2.0.*A .+ B) == (2,2)
        B .= 2.0 .* A .+ B
        @test B == C
    end

    @testset "gbmv!" begin
        n = 100
        x = randn(n)
        A = brand(n,n,1,1)
        y = similar(x)
        y .= Mul(A,x)
        @test Matrix(A)*x ≈ y
        @test all(BLAS.gbmv!('N', n, 1, 1, 1.0, A.data, x, 0.0, copy(y)) .===
                    Broadcast.materialize!(MulAdd(1.0,A,x,0.0,similar(y))) .=== y)
        z = similar(y)
        z .= 2.0.*Mul(A,x) .+ 3.0.*y
        @test 2Matrix(A)*x + 3y ≈ z
        @test all(BLAS.gbmv!('N', n, 1, 1, 2.0, A.data, x, 3.0, copy(y)) .=== z)
        @test MemoryLayout(A') == BandedRows(DenseColumnMajor())
        y .= Mul(A',x)
        @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
        @test all(BLAS.gbmv!('T', n, 1, 1, 1.0, A.data, x, 0.0, copy(y)) .=== y)
        z = similar(y)
        z .= 2.0.*Mul(A',x) .+ 3.0.*y
        @test 2Matrix(A')*x + 3y ≈ z
        @test all(BLAS.gbmv!('T', n, 1, 1, 2.0, A.data, x, 3.0, copy(y)) .=== z)
        @test MemoryLayout(transpose(A)) == BandedRows(DenseColumnMajor())
        y .= Mul(transpose(A),x)
        @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
        @test all(BLAS.gbmv!('T', n, 1, 1, 1.0, A.data, x, 0.0, copy(y)) .=== y)

        n = 100
        x = randn(ComplexF64,n)
        A = brand(ComplexF64,n,n,1,1)
        y = similar(x)
        y .= Mul(A,x)
        @test Matrix(A)*x ≈ y
        @test all(BLAS.gbmv!('N', n, 1, 1, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
        @test LazyArrays.MemoryLayout(A') == ConjLayout(BandedRows(DenseColumnMajor()))
        z = similar(y)
        z .= (2.0+0.0im).*Mul(A,x) .+ (3.0+0.0im).*y
        @test 2Matrix(A)*x + 3y ≈ z
        @test all(BLAS.gbmv!('N', n, 1, 1, 2.0+0.0im, A.data, x, 3.0+0.0im, copy(y)) .=== z)
        y .= Mul(A',x)
        @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
        @test all(BLAS.gbmv!('C', n, 1, 1, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
        @test MemoryLayout(transpose(A)) == BandedRows(DenseColumnMajor())
        y .= Mul(transpose(A),x)
        @test Matrix(transpose(A))*x ≈ transpose(Matrix(A))*x ≈ y
        @test all(BLAS.gbmv!('T', n, 1, 1, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
    end

    @testset "non-blas matvec" begin
        n = 10
        A = brand(Int,n,n,1,1)
        x = rand(Int,n)
        y = similar(x)
        y .= Mul(A,x)
        @test y == Matrix(A)*x
        y .= Mul(A',x)
        @test y == Matrix(A')*x
        y .= Mul(transpose(A),x)
        @test y == Matrix(transpose(A))*x

        A = brand(Int,n,n,1,1) + im*brand(Int,n,n,1,1)
        x = rand(eltype(A),n)
        y = similar(x)
        y .= Mul(A,x)
        @test y == Matrix(A)*x
        y .= Mul(A',x)
        @test y == Matrix(A')*x
        y .= Mul(transpose(A),x)
        @test y == Matrix(transpose(A))*x
    end

    @testset "gbmm!" begin
        n = 10
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        C .= Mul(A,B)
        @test Matrix(C) ≈ Matrix(A)*Matrix(B)
        C .= Mul(A',B)
        @test Matrix(C) ≈ Matrix(A')*Matrix(B)
        C .= Mul(A,B')
        @test Matrix(C) ≈ Matrix(A)*Matrix(B')
        C = brand(n,n,4,4)
        C .= Mul(A,B)
        @test Matrix(C) ≈ Matrix(A)*Matrix(B)
        B = randn(n,n)
        C = similar(B)

        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)
        C .= Mul(B,A)
        @test C ≈ Matrix(B)*Matrix(A)
    end

    @testset "Subarray" begin
        n = 10
        A = brand(n,n,1,1)
        B = view(brand(2n,2n,2,0),Base.OneTo(n),1:n)
        @test BroadcastStyle(typeof(B)) == BandedStyle()

        @test A .+ B isa BandedMatrix
        @test A + B isa BandedMatrix
        @test bandwidths(A .+ B) == (2,1)
        @test A .+ B == Matrix(A) + Matrix(B)
    end
end
