using BandedMatrices, LinearAlgebra, LazyArrays, Test
    import LazyArrays: MemoryLayout

@testset "general" begin
    n = 1000
    A = brand(n,n,1,1)
    B = Matrix{Float64}(undef, n,n)
    B .= exp.(A)
    @test B == exp.(Matrix(A)) == exp.(A)
    @test exp.(A) isa Matrix
    @test A .+ 1 isa Matrix
end

@testset "lmul!/rmul!" begin
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
end

@testset "axpy!" begin
    n = 1000
    A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    C = brand(n,n,3,3)
    @time C .= A .+ B
    @test C == A + B == A .+ B
    @test A + B isa BandedMatrix
    @test A .+ B isa BandedMatrix
    @test bandwidths(A+B) == bandwidths(A.+B) == (2,2)
    @time B .= A .+ B
    @test B == C

    n = 1000
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
    @test LazyArrays.MemoryLayout(A') == LazyArrays.ConjLayout(BandedMatrices.BandedRowMajor())
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
