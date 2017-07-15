# some basic operations

A,B=brand(10,12,2,3),brand(10,12,3,4)

@test full(sparse(A)) ≈ full(A)

@test full(A') ≈ full(A)'
@test full(A.') ≈ full(A).'
@test full((A+im*A)') ≈ (full(A)+im*full(A))'
@test full((A+im*A).') ≈ (full(A)+im*full(A)).'

@test full(A+B) ≈ (full(A)+full(B))
@test full(A-B) ≈ (full(A)-full(B))

@test full(A.*B) ≈ (full(A).*full(B))


# banded * vec

A =brand(10,12,2,3)
v=rand(12)
w=rand(10)
@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w

A=brand(Float64,5,3,2,2)
v=rand(Complex128,3)
w=rand(Complex128,5)
@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w

A=brand(Complex128,5,3,2,2)
v=rand(Complex128,3)
w=rand(Complex128,5)
@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w

v=rand(Float64,3)
w=rand(Float64,5)
@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w

v=rand(Float64,3)
w=rand(Float64,5)
@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w


# test matrix multiplications

# big banded * dense

A=brand(1000,1000,200,300)
B=rand(1000,1000)
@test A*B ≈ full(A)*B
@test B*A ≈ B*full(A)

A=brand(1200,1000,200,300)
B=rand(1000,1000)
C=rand(1200,1200)
@test A*B ≈ full(A)*B
@test C*A ≈ C*full(A)

A=brand(1000,1200,200,300)
B=rand(1200,1200)
C=rand(1000,1000)
@test A*B ≈ full(A)*B
@test C*A ≈ C*full(A)


# banded * banded

for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    @test full(A*B) ≈ full(A)*full(B)
end


## BigFloat

A = brand(5, 5, 1, 2)
B = bzeros(BigFloat,5,5,2,3)
for j = 1:size(B,2), k = colrange(B,j)
    B[k,j]=randn()
end
D = rand(5, 5)
x = BigFloat[1:size(B,1)...]

@test full(A)*full(B) ≈ A*B
@test full(B)*full(A) ≈ B*A
@test full(B)*x ≈ B*x
@test full(B*B) ≈ full(B)*full(B)
@test full(A)*full(D) ≈ A*D
@test full(D)*full(A) ≈ D*A


## negative bands

for A in (brand(3,4,-1,2),brand(5,4,-1,2),
            brand(3,4,2,-1),brand(5,4,2,-1))
    b = rand(size(A,2))
    @test A*b ≈ full(A)*b
end

D = rand(4, 4)
for A in (brand(3,4,1,2),brand(3,4,-1,2),brand(3,4,2,-1)),
    B in (brand(4,4,1,2),brand(4,4,-1,2),brand(4,4,2,-1))
    @test A*B ≈ full(A)*full(B)
    @test A*D ≈ full(A)*full(D)
    @test D*B ≈ full(D)*full(B)
end


## UniformScaling
A = brand(10,10,1,2)

@test full(A+I) ≈ full(A)+I
@test full(I-A) ≈ I-full(A)


# number
@test full(A/2) ≈ full(A)/2

# zero arrays

let b = rand(4)
    for A in (brand(3,4,-1,0),brand(3,4,0,-1),brand(3,4,-1,-1)),
        B in (brand(4,3,1,2),brand(4,3,-1,0),brand(4,3,-1,-1))
        @test full(A) == zeros(3,4)
        @test A*B == zeros(3,3)
        @test A*b == zeros(3)
    end
end


# check that col/rowstop is ≥ 0
A = brand(3,4,-2,2)
@test BandedMatrices.colstop(A, 1) == BandedMatrices.colstop(A, 2) == 0
@test BandedMatrices.colstop(A, 3) == 1

A = brand(3,4,2,-2)
@test BandedMatrices.rowstop(A, 1) == BandedMatrices.rowstop(A, 2) == 0
@test BandedMatrices.rowstop(A, 3) == 1

 # Test for errors in collect

B=brand(10,10,0,4)
@test B*[collect(1.0:10) collect(1.0:10)] ≈ full(B)*[collect(1.0:10) collect(1.0:10)]
