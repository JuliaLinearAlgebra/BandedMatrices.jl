using BandedMatrices, LinearAlgebra, Test, Random
import BandedMatrices: banded_qr!
Random.seed!(0)

@testset "QR tests" begin
    for T in (Float64,ComplexF64,Float32,ComplexF32)
        A=brand(T,10,10,3,2)
        Q,R = qr(A)
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

    @testset "banded Views" begin
        # this test adaptive QR essentially as implemented in InfiniteLinearAlgebra.jl
        A = brand(100,100,1,1)
        V = view(copy(A),1:5,1:6)
        @test qr(V) isa QR{Float64,<:BandedMatrix{Float64}}
        @test qr(V).R ≈ qr(Matrix(V)).R
        @test qr(V).τ ≈ LinearAlgebra.qrfactUnblocked!(Matrix(V)).τ
        @test qr(V).Q ≈ qr(Matrix(V)).Q
        @test Matrix(qr(V)) ≈ V
        B = BandedMatrix(A,(1,2)) # pad
        V = view(copy(B),1:5,1:6)
        @test qr!(V) isa QR{Float64,<:SubArray{Float64,2,<:BandedMatrix{Float64}}}
        τ = Array{Float64}(undef,100)
        B = BandedMatrix(A,(1,2)) # pad
        V = view(B,1:6,1:5)
        F1 = qr(V)
        F2 = banded_qr!(V, view(τ,1:size(V,2)))
        @test F1.factors ≈ F2.factors ≈ B[1:6,1:5]
        @test F1.τ ≈ F2.τ
        F = qr(A)
        @test τ[1:size(V,2)]  ≈ F.τ[1:5]

        lmul!(F1.Q', view(B,1:6,6:7))
        @test B[1:5,6:7] ≈ F.factors[1:5,6:7]
        banded_qr!(view(B,6:100,6:100), view(τ,6:100))
        τ[end] = 0
        @test B ≈ F.factors
        @test τ ≈ F.τ
    end

    @testset "banded_lmul! times degenerate" begin
        B = brand(10,10,1,1)
        Q,R = qr(B)
        @test lmul!(Q', BandedMatrix(B,(size(B,1),2))) ≈ Q'*B
        @test lmul!(Q', view(BandedMatrix(B,(size(B,1),2)),:,4:10)) ≈ Q'*B[:,4:10]
    end

    @testset "rmul!" begin
        for T in (Float64,ComplexF64,Float32,ComplexF32)
            A = brand(T,10,10,3,2)
            Q,R = qr(A)
            B = randn(T, 2, 10)
            @test rmul!(copy(B), Q') ≈ B*Q'
            @test rmul!(copy(B), Q) ≈ B*Q
        end
    end
end
