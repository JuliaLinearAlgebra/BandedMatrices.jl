A=brand(10,10,3,2)
Q,R=qr(A)


@test_approx_eq full(Q)*full(R) A

b=rand(10)
@test_approx_eq A_mul_B!(similar(b),Q,At_mul_B!(similar(b),Q,b)) b


for j=1:size(A,2)
    @test_approx_eq At_mul_B(Q,A[:,j]) R[:,j]
end


A=brand(14,10,3,2)

Q,R=qr(A)


for k=1:size(Q,1),j=1:size(Q,2)
    @test_approx_eq Q[k,j] full(Q)[k,j]
end

@test_approx_eq full(Q)*full(R) A



A=brand(10,14,3,2)

Q,R=qr(A)


for k=1:size(Q,1),j=1:size(Q,2)
    @test_approx_eq Q[k,j] full(Q)[k,j]
end

@test_approx_eq full(Q)*full(R) A
A=brand(100,100,3,4)
Q,R=qr(A)
b=rand(100)
@test_approx_eq R\(Q'*b) full(A)\b

println("Large dimension QR solve")
A=brand(100000,100000,3,4)
b=rand(size(A,1))
@time Q,R=qr(A)
@time R\(Q'*b)

println("Time should be   0.030149 seconds (23 allocations: 17.548 MB)")
println("Time should be   0.008210 seconds (29 allocations: 8.012 MB)")


A=brand(2000,2000,400,400)
b=rand(size(A,1))
@time Q,R=qr(A)
@time R\(Q'*b)


println("Time should be   0.401671 seconds (24 allocations: 36.665 MB)")
println("Time should be   0.009276 seconds (30 allocations: 12.262 MB)")
