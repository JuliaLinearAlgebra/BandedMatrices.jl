using BandedMatrices, LinearAlgebra, Test

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

        A=brand(T,100,100,3,4) + 10I
        Q,R=qr(A)
        b=rand(T,100)
        @test R\(Q'*b) ≈ qr(A)\b ≈ Matrix(A)\b

        A=brand(T,102,100,3,4)
        Q,R=qr(A)
        b=rand(T,102)
        qr(A)\b ≈ Matrix(A)\b
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
    end
end
