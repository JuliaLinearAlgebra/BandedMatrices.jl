versioninfo()

#gc_enable(false)

using BandedMatrices

A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)
C = rand(1000, 1000)
D = rand(10000, 1000)
tD = D'
v=rand(10000)
w=rand(1000)

print("A*A:")
A*A;
@time for k=1:100
    A*A
end

print("A*D:")
A*D;
@time for k=1:10
    A*D
end

print("D*A:")
tD*A;
@time for k=1:10
    tD*A
end

print("B*B:")
B*B;
@time for k=1:10
    B*B
end

print("B*C:")
B*C;
@time for k=1:10
    B*C
end

print("C*B:")
C*B;
@time for k=1:10
    C*B
end

print("A*v:")
A*v;
@time for k=1:1000
    A*v
end

print("B*w:")
B*w;
@time for k=1:100
    B*w
end

#gc_enable(true)

function julia(n, b1, b2)
    A = brand(n, n, b1, b2)
    BandedMatrices.banded_generic_matmatmul!(BandedMatrix(Float64,n, n, 2*b1, 2*b2), A, A)
end

function blas(n, b1, b2)
    A = brand(n, n, b1, b2)
    BandedMatrices.gbmm!(one(Float64), A, A, zero(Float64), BandedMatrix(Float64,n, n, 2*b1, 2*b2))
end

function julia(n, b1, b2)
    A = brand(n, n, b1, b2)
    D = rand(n, 2)
    BandedMatrices.banded_generic_matmatmul!(Matrix{Float64}(n, 2), A, D)
end

function blas(n, b1, b2)
    A = brand(n, n, b1, b2)
    D = rand(n, 2)
    BandedMatrices.gbmm!(one(Float64), A, D, zero(Float64), Matrix{Float64}(n, 2))
end

@time julia(100, 1, 1);
@time blas(100, 1, 1);
@time julia(100, 50, 50);
@time blas(100, 50, 50);
@time julia(100, 75, 75);
@time blas(100, 75, 75);
@time julia(1000, 2, 3);
@time blas(1000, 2, 3);
@time julia(1000, 50, 50);
@time blas(1000, 50, 50);
@time julia(1000, 200, 300);
@time blas(1000, 200, 300);
@time julia(10000, 2, 3);
@time blas(10000, 2, 3);
@time julia(10000, 50, 50);
@time blas(10000, 50, 50);
@time julia(10000, 200, 300);
@time blas(10000, 200, 300);
@time julia(100000, 2, 3);
@time blas(100000, 2, 3);
@time julia(100000, 50, 50);
@time blas(100000, 50, 50);
@time julia(100000, 200, 300);
@time blas(100000, 200, 300);


# println("Large dimension QR solve")
# A=brand(100000,100000,3,4)
# b=rand(size(A,1))
# @time Q,R=qr(A)
# @time R\(Q'*b)

# println("Time should be   0.030149 seconds (23 allocations: 17.548 MB)")
# println("Time should be   0.008210 seconds (29 allocations: 8.012 MB)")


# A=brand(2000,2000,400,400)
# b=rand(size(A,1))
# @time Q,R=qr(A)
# @time R\(Q'*b)


# println("Time should be   0.401671 seconds (24 allocations: 36.665 MB)")
# println("Time should be   0.009276 seconds (30 allocations: 12.262 MB
