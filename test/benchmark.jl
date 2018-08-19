versioninfo()

#gc_enable(false)

using BandedMatrices

A = brand(10000,10000,2,3)
B = brand(1000,1000,200,300)
E = brand(10000,10000,-2,2)
C = rand(1000, 1000)
D = rand(10000, 1000)
V = view(brand(10000,10000,2,3), :, :)
tD = D'
v=rand(10000)
w=rand(1000)

print("A*A:")
A*A;
@time for k=1:100
    A*A
end

print("V*V:")
V*V;
@time for k=1:100
    V*V
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

print("A*E:")
A*E
@time for k=1:10
    A*E
end

print("E*D:")
E*D
@time for k=1:10
    E*D
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

# benchmark pure julia routine vs BLAS routine

# Banded * Dense

julia(A, D) = BandedMatrices._banded_generic_matmatmul!(Matrix{Float64}(size(A, 1), size(D, 2)), 'N', 'N', A, D)
blas(A, D) = BandedMatrices.gbmm!('N', 'N', one(Float64), A, D, zero(Float64), Matrix{Float64}(size(A, 1), size(D, 2)))
dense(A, D) = Array(A)*Array(D)

for n in [100, 1000, 10000, 100000]
    for b in [1, 50, 200, 500]
        A, D = brand(n, n, b, b), rand(n, 2)
        println("n = $n, b = $b")
        if b == 1 && n ≤ 10000
            print("dense")
            @time dense(A, D)
        end
        print("julia")
        @time julia(A, D);
        print("blas ")
        @time blas(A, D);
    end
end

# Banded * Banded

julia(A) = BandedMatrices._banded_generic_matmatmul!(
    BandedMatrix{Float64}(undef, size(A), (2*bandwidth(A, 1), 2*bandwidth(A, 2))), 'N', 'N', A, A)
blas(A) = BandedMatrices.gbmm!(
    'N', 'N', one(Float64), A, A, zero(Float64), BandedMatrix{Float64}(undef, size(A), (2*bandwidth(A, 1), 2*bandwidth(A, 2))))
dense(A) = (B=Array(A); B*B)

for n in [100, 1000, 10000]
    for b in [1, 50, 200, 500]
        A = brand(n, n, b, b)
        println("n = $n, b = $b")
        if b == 1 && n ≤ 1000
            print("dense")
            @time dense(A)
        end
        print("julia")
        @time julia(A);
        print("blas ")
        @time blas(A);
    end
end

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

nothing
