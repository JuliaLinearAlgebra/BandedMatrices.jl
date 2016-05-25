versioninfo()

using BandedMatrices, Base.Test

include("test_indexing.jl")
include("test_bandedlu.jl")

A,B=brand(10,12,2,3),brand(10,12,3,4)


@test_approx_eq full(sparse(A)) full(A)

@test_approx_eq full(A') full(A)'
@test_approx_eq full(A.') full(A).'
@test_approx_eq full((A+im*A)') (full(A)+im*full(A))'
@test_approx_eq full((A+im*A).') (full(A)+im*full(A)).'

@test_approx_eq full(A+B) (full(A)+full(B))
@test_approx_eq full(A-B) (full(A)-full(B))

@test_approx_eq full(A.*B) (full(A).*full(B))

C,D=brand(10,10,2,3),brand(12,12,3,4)

@test_approx_eq full(C*A) full(C)*full(A)
@test_approx_eq full(A*D) full(A)*full(D)


v=rand(12)
w=rand(10)

@test_approx_eq A*v full(A)*v
@test_approx_eq A'*w full(A)'*w


A=brand(Float64,5,3,2,2)
v=rand(Complex128,3)
@test_approx_eq A*v full(A)*v

A=brand(Complex128,5,3,2,2)
v=rand(Complex128,3)
@test_approx_eq A*v full(A)*v

v=rand(Float64,3)
@test_approx_eq A*v full(A)*v



 # Test for errors in collect
A=brand(12,10,2,3)
collect(BandedMatrices.eachbandedindex(A))
for (k,j) in BandedMatrices.eachbandedindex(A)
     A[k,j]
end


A=brand(1000,1000,200,300)
B=rand(1000,1000)
@test_approx_eq A*B full(A)*B

A=brand(1200,1000,200,300)
B=rand(1000,1000)
@test_approx_eq A*B full(A)*B

A=brand(1000,1200,200,300)
B=rand(1200,1200)
@test_approx_eq A*B full(A)*B



B=brand(10,10,0,4)
@test_approx_eq B*[collect(1.0:10) collect(1.0:10)] full(B)*[collect(1.0:10) collect(1.0:10)]


@which B*[collect(1.0:10) collect(1.0:10)]


A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)
v=rand(10000)
w=rand(1000)

show(brand(10,10,3,3))
show(brand(100,80,3,2))
println()

gc_enable(false)

A*v
@time for k=1:100
    A*v
end
println("Time should be   0.007245 seconds (400 allocations: 7.639 MB)")
@time for k=1:100
    B*w
end
println("Time should be   0.017208 seconds (300 allocations: 792.188 KB)")

for n in (10,50), m in (12,50), Al in (0,1,2,30), Au in (0,1,2,30)
    A=brand(n,m,Al,Au)
    kr,jr=3:10,5:12
    @test_approx_eq full(A[kr,jr]) full(A)[kr,jr]
end


for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    @test_approx_eq full(A*B) full(A)*full(B)
end

A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)


@time for k=1:10
    A*A
end

println("Time should be   0.010083 seconds (30 allocations: 8.393 MB)")

@time for k=1:10
    B*B
end

println("Time should be   0.644119 seconds (30 allocations: 76.371 MB)")

gc_enable(true)




## Banded Matrix of Banded Matrix


A=BandedMatrix(BandedMatrix{Float64},1,2,0,1)
A[1,1]=beye(1,1,0,1)
A[1,2]=bzeros(1,2,0,1)
A[1,2][1,1]=-1/3
A[1,2][1,2]=1/3
B=BandedMatrix(BandedMatrix{Float64},2,1,1,1)
B[1,1]=0.2beye(1,1,0,1)
B[2,1]=bzeros(2,1,1,0)
B[2,1][1,1]=-2/30
B[2,1][2,1]=1/3

@test_approx_eq (A*B)[1,1] 1/3



## Bug

# BandedMatrices.eachbandedindex(BandedMatrix(Float64,1,2,2,1))|>collect
