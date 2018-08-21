using BandedMatrices, LinearAlgebra, Test

@testset "lmul!/rmul!" begin
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    B .= 2.0.*A

    @test B ==  2A
    A .= 2.0.*A
    @test A == B

    n = 1000
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    B .= 2.0.*A

    @test B ==  2A
    A .= 2.0.*A
    @test A == B

    n = 1000
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    B .= A.*2.0

    @test B ==  2A
    A .= A.*2.0
    @test A == B
end

@testset "axpy!" begin
    n = 1000
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    C = brand(n,n,3,3)
    @time C .= A .+ B
    @test C == A + B
    @time B .= A .+ B
    @test B == C


    n = 1000
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    C = brand(n,n,3,3)

    C .= 2.0 .* A .+ B
    @test C == 2A+B
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
    @test all(BLAS.gbmv!('N', n, 1, 1, 1.0, A.data, x, 0.0, copy(y)) .=== y)
    z = similar(y)
    z .= 2.0.*Mul(A,x) .+ 3.0.*y
    @test 2Matrix(A)*x + 3y ≈ z
    @test all(BLAS.gbmv!('N', n, 1, 1, 2.0, A.data, x, 3.0, copy(y)) .=== z)
    @test MemoryLayout(A') == BandedMatrices.BandedRowMajor()
    y .= Mul(A',x)
    @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
    @test all(BLAS.gbmv!('T', n, 1, 1, 1.0, A.data, x, 0.0, copy(y)) .=== y)
    z = similar(y)
    z .= 2.0.*Mul(A',x) .+ 3.0.*y
    @test 2Matrix(A')*x + 3y ≈ z
    @test all(BLAS.gbmv!('T', n, 1, 1, 2.0, A.data, x, 3.0, copy(y)) .=== z)
    @test MemoryLayout(transpose(A)) == BandedMatrices.BandedRowMajor()
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
    @test MemoryLayout(A') == LazyArrays.ConjLayout(BandedMatrices.BandedRowMajor())
    z = similar(y)
    z .= (2.0+0.0im).*Mul(A,x) .+ (3.0+0.0im).*y
    @test 2Matrix(A)*x + 3y ≈ z
    @test all(BLAS.gbmv!('N', n, 1, 1, 2.0+0.0im, A.data, x, 3.0+0.0im, copy(y)) .=== z)
    y .= Mul(A',x)
    @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
    @test all(BLAS.gbmv!('C', n, 1, 1, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
    @test MemoryLayout(transpose(A)) == BandedMatrices.BandedRowMajor()
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
