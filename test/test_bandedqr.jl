using BandedMatrices, Compat.Test

if VERSION < v"0.7-"
    const mul! = Base.A_mul_B!
end
@testset "QR tests" begin
    for T in (Float64,ComplexF64,Float32,ComplexF32)
        A=brand(T,10,10,3,2)
        Q,R=qr(A)
        @test Matrix(Q)*Matrix(R) ≈ A
        b=rand(T,10)
        @test mul!(similar(b),Q,BandedMatrices.Ac_mul_B!(similar(b),Q,b)) ≈ b
        for j=1:size(A,2)
            @test BandedMatrices.Ac_mul_B(Q,A[:,j]) ≈ R[:,j]
        end
        A=brand(T,14,10,3,2)

        Q,R=qr(A)


        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        @test Matrix(Q)*Matrix(R) ≈ A

        A=brand(T,10,14,3,2)

        Q,R=qr(A)


        for k=1:size(Q,1),j=1:size(Q,2)
            @test Q[k,j] ≈ Matrix(Q)[k,j]
        end

        @test Matrix(Q)*Matrix(R) ≈ A
        A=brand(T,100,100,3,4)
        Q,R=qr(A)
        b=rand(T,100)
        @test R\(Q'*b) ≈ Matrix(A)\b
    end


    A=brand(10,10,3,2)
    b=rand(ComplexF64,10)
    Q,R=qr(A)
    @test R\(Q'*b) ≈ Matrix(A)\b

    A=brand(ComplexF64,10,10,3,2)
    b=rand(10)
    Q,R=qr(A)
    @test R\(Q'*b) ≈ Matrix(A)\b
end
