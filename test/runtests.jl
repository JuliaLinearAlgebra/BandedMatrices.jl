using BandedMatrices, Base.Test

include("test_gbmm.jl")
include("test_indexing.jl")
include("test_bandedlu.jl")
include("test_bandedqr.jl")
include("test_symbanded.jl")


A,B=brand(10,12,2,3),brand(10,12,3,4)


@test full(sparse(A)) ≈ full(A)

@test full(A') ≈ full(A)'
@test full(A.') ≈ full(A).'
@test full((A+im*A)') ≈ (full(A)+im*full(A))'
@test full((A+im*A).') ≈ (full(A)+im*full(A)).'

@test full(A+B) ≈ (full(A)+full(B))
@test full(A-B) ≈ (full(A)-full(B))

@test full(A.*B) ≈ (full(A).*full(B))

C,D=brand(10,10,2,3),brand(12,12,3,4)

@test full(C*A) ≈ full(C)*full(A)
@test full(A*D) ≈ full(A)*full(D)


v=rand(12)
w=rand(10)

@test A*v ≈ full(A)*v
@test A'*w ≈ full(A)'*w


A=brand(Float64,5,3,2,2)
v=rand(Complex128,3)
@test A*v ≈ full(A)*v

A=brand(Complex128,5,3,2,2)
v=rand(Complex128,3)
@test A*v ≈ full(A)*v

v=rand(Float64,3)
@test A*v ≈ full(A)*v



 # Test for errors in collect
A=brand(12,10,2,3)

for j = 1:size(A,2), k = colrange(A,j)
     A[k,j]
end


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



B=brand(10,10,0,4)
@test B*[collect(1.0:10) collect(1.0:10)] ≈ full(B)*[collect(1.0:10) collect(1.0:10)]


# TODO: test show function
# show(brand(10,10,3,3))
# show(brand(100,80,3,2))
# println()



for n in (10,50), m in (12,50), Al in (0,1,2,30), Au in (0,1,2,30)
    A=brand(n,m,Al,Au)
    kr,jr=3:10,5:12
    @test full(A[kr,jr]) ≈ full(A)[kr,jr]
end

# test matrix multiplications

for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    α,β,T=0.123,0.456,Float64
    C=brand(Float64,n,m,A.l+B.l,A.u+B.u)

    exC=α*full(A)*full(B)+β*full(C)
    BandedMatrices.gbmm!(α,A,B,β,C)

    @test full(exC) ≈ full(C)
end



for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    @test full(A*B) ≈ full(A)*full(B)
end

## Banded Matrix of Banded Matrix

BandedMatrixWithZero = Union{BandedMatrix{Float64}, UniformScaling}
# need to define the concept of zero
Base.zero(::Type{BandedMatrixWithZero}) = 0*I

A=BandedMatrix(BandedMatrixWithZero,1,2,0,1)
A[1,1]=beye(1,1,0,1)
A[1,2]=bzeros(1,2,0,1)
A[1,2][1,1]=-1/3
A[1,2][1,2]=1/3
B=BandedMatrix(BandedMatrixWithZero,2,1,1,1)
B[1,1]=0.2beye(1,1,0,1)
B[2,1]=bzeros(2,1,1,0)
B[2,1][1,1]=-2/30
B[2,1][2,1]=1/3

@test (A*B)[1,1][1,1] ≈ 1/3




## BigFloat


B=bzeros(BigFloat,5,5,2,3)
    for j = 1:size(B,2), k = colrange(B,j)
        B[k,j]=randn()
    end

x=BigFloat[1:size(B,1)...]





@test full(B)*x ≈ B*x
@test full(B*B) ≈ full(B)*full(B)



A = brand(10,10,1,2)
B = brand(20,20,1,2)

@test isa(view(B,1:10,1:10),BandedMatrices.BandedSubBandedMatrix{Float64})

B2 = copy(B)
@test (2.0A+B[1:10,1:10]) ≈ Base.axpy!(2.0,A,view(B2,1:10,1:10))
@test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

A2 = copy(A)
@test (2.0B[1:10,1:10]+A) ≈ Base.axpy!(2.0,view(B,1:10,1:10),A2)
@test (2.0B[1:10,1:10]+A) ≈ A2

A = brand(20,20,1,2)
B = brand(20,20,1,2)

@test isa(view(B,1:10,1:10),BandedMatrices.BandedSubBandedMatrix{Float64})

B2 = copy(B)
@test (2.0A[1:10,1:10]+B[1:10,1:10]) ≈ Base.axpy!(2.0,view(A,1:10,1:10),view(B2,1:10,1:10))
@test (2.0A[1:10,1:10]+B[1:10,1:10]) ≈ B2[1:10,1:10]

B2 = copy(B)
@test (2.0A[1:10,:]+B[1:10,:]) ≈ Base.axpy!(2.0,view(A,1:10,:),view(B2,1:10,:))
@test (2.0A[1:10,:]+B[1:10,:]) ≈ B2[1:10,:]

B2 = copy(B)
@test (2.0A[:,1:10]+B[:,1:10]) ≈ Base.axpy!(2.0,view(A,:,1:10),view(B2,:,1:10))
@test (2.0A[:,1:10]+B[:,1:10]) ≈ B2[:,1:10]


B2 = copy(B)
@test (2.0A[:,:]+B[:,:]) ≈ Base.axpy!(2.0,view(A,:,:),view(B2,:,:))
@test (2.0A[:,:]+B[:,:]) ≈ B2[:,:]



A = brand(10,10,1,2)
B = brand(20,10,1,2)

B2 = copy(B)
@test (2.0A+B[1:10,1:10]) ≈ Base.axpy!(2.0,A,view(B2,1:10,:))
@test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

A2 = copy(A)
@test (2.0B[1:10,1:10]+A) ≈ Base.axpy!(2.0,view(B,1:10,:),A2)
@test (2.0B[1:10,1:10]+A) ≈ A2


A = brand(10,10,1,2)
B = brand(10,20,1,2)

B2 = copy(B)
@test (2.0A+B[1:10,1:10]) ≈ Base.axpy!(2.0,A,view(B2,:,1:10))
@test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

A2 = copy(A)
@test (2.0B[1:10,1:10]+A) ≈ Base.axpy!(2.0,view(B,:,1:10),A2)
@test (2.0B[1:10,1:10]+A) ≈ A2



## UniformScalin g
A = brand(10,10,1,2)

@test full(A+I) ≈ full(A)+I
@test full(I-A) ≈ I-full(A)


@test full(A/2) ≈ full(A)/2


## Test StridedMatrix
A=brand(10,10,1,2)
v=rand(20)

@test A*view(v,1:10) ≈ full(A)*v[1:10]
@test A*view(v,1:2:20) ≈ full(A)*v[1:2:20]

M=rand(20,20)
@test A*view(M,1:10,1:10) ≈ full(A)*M[1:10,1:10]
@test A*view(M,1:2:20,1:2:20) ≈ full(A)*M[1:2:20,1:2:20]


A=brand(10,10,1,2)
B=brand(20,10,1,2)
C=brand(10,20,1,2)
D=brand(20,20,1,2)
M=rand(10,10)
V=view(rand(20,20),1:2:20,1:2:20)


for S in (view(A,:,:),view(B,1:10,:),view(C,:,1:10),view(D,1:10,1:10),
            view(D,2:11,1:10),view(D,1:10,2:11),view(D,11:20,11:20))
    @test A*S ≈ full(A)*full(S)
    @test S*A ≈ full(S)*full(A)
    @test S*S ≈ full(S)*full(S)

    @test M*S ≈ M*full(S)
    @test S*M ≈ full(S)*M

    @test V*S ≈ full(V)*full(S)
    @test S*V ≈ full(S)*full(V)
end


## negative bands

for A in (brand(3,4,-1,2),brand(5,4,-1,2),
            brand(3,4,2,-1),brand(5,4,2,-1))
    b = rand(size(A,2))
    @test A*b ≈ full(A)*b
end


for A in (brand(3,4,1,2),brand(3,4,-1,2),brand(3,4,2,-1)),
    B in (brand(4,4,1,2),brand(4,4,-1,2),brand(4,4,2,-1))
    @test A*B ≈ full(A)*full(B)
end

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

# test trivial convert routines

A = brand(3,4,1,2)

BandedMatrix{Float64}(A)
@test isa(BandedMatrix{Float64}(A),BandedMatrix{Float64})
@test isa(AbstractMatrix{Float64}(A),BandedMatrix{Float64})
@test isa(AbstractArray{Float64}(A),BandedMatrix{Float64})
@test isa(BandedMatrix(A),BandedMatrix{Float64})
@test isa(AbstractMatrix(A),BandedMatrix{Float64})
@test isa(AbstractArray(A),BandedMatrix{Float64})
@test isa(BandedMatrix{Complex32}(A),BandedMatrix{Complex32})
@test isa(AbstractMatrix{Complex32}(A),BandedMatrix{Complex32})
@test isa(AbstractArray{Complex32}(A),BandedMatrix{Complex32})


# Test dense overrides
A = rand(10,11)
@test bandwidths(A) == (9,10)
A = rand(10)
@test bandwidths(A) == (9,0)
@test bandwidths(A') == (0,9)
