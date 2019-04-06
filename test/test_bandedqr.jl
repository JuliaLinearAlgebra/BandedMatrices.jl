using BandedMatrices, MatrixFactorizations, LinearAlgebra, Test, Random

Random.seed!(0)


@testset "QR tests" begin
    for T in (Float64,ComplexF64,Float32,ComplexF32)
        A=brand(T,10,10,3,2)
        Q,R=qr(A)
        @test Matrix(Q)*Matrix(R) ≈ A
        b=rand(T,10)
        @test mul!(similar(b),Q,mul!(similar(b),Q',b)) ≈ b
        for j=1:size(A,2)
            @test Q' * A[:,j] ≈ R[:,j]
        end
        A=brand(T,14,10,3,2)

        Q,R=qr(A)
        @test Matrix(Q)*Matrix(R) ≈ A

        for k=1:size(A,1),j=1:size(A,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,10,14,3,2)
        Q,R=qr(A)
        @test Matrix(Q)*Matrix(R) ≈ A

        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,100,100,3,4)
        @test qr(A).factors ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).factors
        @test qr(A).τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ
        b=rand(T,100)
        @test qr(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test qr(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch qr(A) \ randn(3)
        @test_throws DimensionMismatch qr(A).Q'randn(3)

        A=brand(T,102,100,3,4)
        @test qr(A).factors ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).factors
        @test qr(A).τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ
        b=rand(T,102)
        @test qr(A)\b ≈ Matrix(A)\b
        b=rand(T,102,2)
        @test qr(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch qr(A) \ randn(3)
        @test_throws DimensionMismatch qr(A).Q'randn(3)

        A=brand(T,100,102,3,4)
        @test qr(A).factors ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).factors
        @test qr(A).τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ
        b=rand(T,100)
        @test_broken qr(A)\b ≈ Matrix(A)\b

        A = Tridiagonal(randn(T,99), randn(T,100), randn(T,99))
        @test qr(A).factors ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).factors
        @test qr(A).τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(A)).τ
        b=rand(T,100)
        @test qr(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test qr(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch qr(A) \ randn(3)
        @test_throws DimensionMismatch qr(A).Q'randn(3)
    end

    @testset "Mixed types" begin
        A=brand(10,10,3,2)
        b=rand(ComplexF64,10)
        Q,R=qr(A)
        @test R\(Q'*b) ≈ qr(A)\b ≈ Matrix(A)\b


        A=brand(ComplexF64,10,10,3,2)
        b=rand(10)
        Q,R=qr(A)
        @test R\(Q'*b) ≈ qr(A)\b ≈ Matrix(A)\b

        A = BandedMatrix{Int}(undef, (2,1), (4,4))
        A.data .= 1:length(A.data)
        Q, R = qr(A)
        @test Q*R ≈ A
    end
end


@testset "QL tests" begin
    for T in (Float64,ComplexF64,Float32,ComplexF32)
        A=brand(T,10,10,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A
        b=rand(T,10)
        @test mul!(similar(b),Q,mul!(similar(b),Q',b)) ≈ b
        for j=1:size(A,2)
            @test Q' * A[:,j] ≈ L[:,j]
        end

        A=brand(T,14,10,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test_broken Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(A,1),j=1:size(A,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,10,14,3,2)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,10,14,3,6)
        Q,L=ql(A)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        @test Matrix(Q)*Matrix(L) ≈ A

        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        A=brand(T,100,100,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test ql(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)

        A=brand(T,102,100,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,102)
        @test_broken ql(A)\b ≈ Matrix(A)\b
        b=rand(T,102,2)
        @test_broken ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)

        A=brand(T,100,102,3,4)
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test_broken ql(A)\b ≈ Matrix(A)\b

        A = Tridiagonal(randn(T,99), randn(T,100), randn(T,99))
        @test ql(A).factors ≈ ql!(Matrix(A)).factors
        @test ql(A).τ ≈ ql!(Matrix(A)).τ
        b=rand(T,100)
        @test ql(A)\b ≈ Matrix(A)\b
        b=rand(T,100,2)
        @test ql(A)\b ≈ Matrix(A)\b
        @test_throws DimensionMismatch ql(A) \ randn(3)
        @test_throws DimensionMismatch ql(A).Q'randn(3)
    end

    @testset "lmul!/rmul!" begin
        A = brand(100,100,3,4)
        Q,R = qr(A)
        x = randn(100)
        b = randn(100,2)
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100)
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')

        A = brand(100,100,3,4)
        Q,L = ql(A)
        x = randn(100)
        b = randn(100,2)
        @test lmul!(Q, copy(x)) ≈ Matrix(Q)*x
        @test lmul!(Q, copy(b)) ≈ Matrix(Q)*b
        @test lmul!(Q', copy(x)) ≈ Matrix(Q)'*x
        @test lmul!(Q', copy(b)) ≈ Matrix(Q)'*b
        c = randn(2,100)
        @test rmul!(copy(c), Q) ≈ c*Matrix(Q)
        @test rmul!(copy(c), Q') ≈ c*Matrix(Q')
    end

    @testset "Mixed types" begin
        A=brand(10,10,3,2)
        b=rand(ComplexF64,10)
        Q,L=ql(A)
        @test L\(Q'*b) ≈ ql(A)\b ≈ Matrix(A)\b
        @test Q*L ≈ A

        A=brand(ComplexF64,10,10,3,2)
        b=rand(10)
        Q,L=ql(A)
        @test Q*L ≈ A
        @test L\(Q'*b) ≈ ql(A)\b ≈ Matrix(A)\b

        A = BandedMatrix{Int}(undef, (2,1), (4,4))
        A.data .= 1:length(A.data)
        Q, L = ql(A)
        @test_broken Q*L ≈ A
    end
end
