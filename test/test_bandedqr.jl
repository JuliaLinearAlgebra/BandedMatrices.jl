using BandedMatrices, Base.Test

for T in (Float64,Complex128,Float32,Complex64)
    A=brand(T,10,10,3,2)
    Q,R=qr(A)
    @test full(Q)*full(R) ≈ A
    b=rand(T,10)
    @test A_mul_B!(similar(b),Q,Ac_mul_B!(similar(b),Q,b)) ≈ b
    for j=1:size(A,2)
        @test Ac_mul_B(Q,A[:,j]) ≈ R[:,j]
    end
    A=brand(T,14,10,3,2)

    Q,R=qr(A)


    for k=1:size(Q,1),j=1:size(Q,2)
        @test Q[k,j] ≈ full(Q)[k,j]
    end

    @test full(Q)*full(R) ≈ A

    A=brand(T,10,14,3,2)

    Q,R=qr(A)


    for k=1:size(Q,1),j=1:size(Q,2)
        @test Q[k,j] ≈ full(Q)[k,j]
    end

    @test full(Q)*full(R) ≈ A
    A=brand(T,100,100,3,4)
    Q,R=qr(A)
    b=rand(T,100)
    @test R\(Q'*b) ≈ full(A)\b
end


A=brand(10,10,3,2)
b=rand(Complex128,10)
Q,R=qr(A)
@test R\(Q'*b) ≈ full(A)\b

A=brand(Complex128,10,10,3,2)
b=rand(10)
Q,R=qr(A)
@test R\(Q'*b) ≈ full(A)\b
