versioninfo()

using BandedMatrices, Base.Test

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


A*v
@time for k=1:100
    A*v
end

@time for k=1:100
    B*w
end


for n in (10,50), m in (12,50), Al in (0,1,2,30), Au in (0,1,2,30)
    A=brand(n,m,Al,Au)
    kr,jr=3:10,5:12
    @test_approx_eq full(A[kr,jr]) full(A)[kr,jr]
end

for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    println("$n,$ν,$m,$Al,$Au,$Bl,$Bu")
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    @test_approx_eq full(A*B) full(A)*full(B)
end



n,ν,m,Al,Au,Bl,Bu=50,5,5,0,0,30,0
A=brand(n,ν,Al,Au)
B=brand(ν,m,Bl,Bu)
@test_approx_eq full(A*B) full(A)*full(B)
C=A*B
C.data
@which A*B
α,β,T=1.0,0.0,Float64
    C=bzeros(Float64,n,m,A.l+B.l,A.u+B.u)
    a=pointer(A.data)
    b=pointer(B.data)
    c=pointer(C.data)
    sta=max(1,stride(A.data,2))
    stb=max(1,stride(B.data,2))
    stc=max(1,stride(C.data,2))

    sz=sizeof(T)

    j=5
    Amid_Bbot_Cmid_gbmv!(α,β,
                                   n,ν,m,j,
                                   sz,
                                   a,A.l,A.u,sta,
                                   b,B.l,B.u,stb,
                                   c,C.l,C.u,stc)
   C





BandedMatrices.gbmm!(α,A,B,β,C)
a=pointer(A.data)
b=pointer(B.data)
c=pointer(C.data)
sta=max(1,stride(A.data,2))
stb=max(1,stride(B.data,2))
stc=max(1,stride(C.data,2))
j=5
@assert ((j-1)*stb + B.l+B.u+1-(j-ν+B.l) ≤ length(B.data))
@assert ((j-1)*stc + n-j+C.u+1  ≤ length(C.data))
gbmv!('N', n-j+C.u+1,
        A.l+A.u, 0,
        α,
        a+sz*(j-B.u-1)*sta, B.l+B.u+1-(j-ν+B.l), sta,
        b+sz*(j-1)*stb, β,
        c+sz*(j-1)*stc)

C


C.m




(j-1)*stc + n-j+C.u+1

length(C.data)

(j-1)*stc

n-j+C.u+1


C.l

C


β

A.l+A.u

B.l+B.u+1-(j-ν+B.l)

n-j+C.u+1
(j-B.u-1)
max(ν-B.l+1,n-C.l+1,C.u+1),min(m,n+C.u,B.u+ν)


ν-B.l+1,min(n,m)-C.l+1,C.u+1

min(m,n+C.u,B.u+ν)

α,β=1.0,0.0
T=Float64
n,ν=size(A)
m=size(B,2)

min(m,n+C.u,B.u+ν)


A
B
@assert n==size(C,1)
@assert ν==size(B,1)
@assert m==size(C,2)


a=pointer(A.data)
b=pointer(B.data)
c=pointer(C.data)
sta=max(1,stride(A.data,2))
stb=max(1,stride(B.data,2))
stc=max(1,stride(C.data,2))
sz=sizeof(T)
import BandedMatrices: gbmv!

# Multiply columns j where B[1,j]≠0: A is at 1,1 and C[1,j]≠0
cols=1:min(B.u+1,m)
print(cols,",")
for j=cols
    @assert ((j-1)*stb+B.u-j+1 + min(B.l+j,ν) ≤ length(B.data))
    @assert ((j-1)*stc+C.u-j+1 +min(C.l+j,n) ≤ length(C.data))
    gbmv!('N',min(C.l+j,n),
           A.l, A.u,
           α,
           a, min(B.l+j,ν), sta,
           b+sz*((j-1)*stb+B.u-j+1), β,
           c+sz*((j-1)*stc+C.u-j+1))
end

C

if last(cols)==m
    return C
end

# Multiply columns j where B[k,j]=0 for k<p=(j-B.u-1), A is at 1,1+p and C[1,j]≠0
# j ≤ ν + B.u since then 1+p ≤ ν, so inside the columns of A
cols=B.u+2:min(C.u+1,m,ν+B.u)
print(cols,",")
for j=cols
    p=j-B.u-1
    @assert ((j-1)*stb + min(B.l+B.u+1,ν-p) ≤ length(B.data))
    @assert ((j-1)*stc+C.u-j+1 +min(C.l+j,n) ≤ length(C.data))
    gbmv!('N', min(C.l+j,n),
            A.l+p, A.u-p,
            α,
            a+sz*p*sta, min(B.l+B.u+1,ν-p), sta,
            b+sz*(j-1)*stb, β,
            c+sz*((j-1)*stc+C.u-j+1))
end
if last(cols)==m
    return C
end

ν-B.l

cols=C.u+2:min(n-C.l,m,ν+B.u)
print(cols,",")
j=2
@assert ((j-1)*stb + B.l+B.u+1 ≤ length(B.data))
@assert ((j-1)*stc + C.l+C.u+1 ≤ length(C.data))
gbmv!('N', B.l+C.u+1,
        A.l+A.u, 0,
        α,
        a+sz*(j-B.u-1)*sta, B.l+B.u+1, sta,
        b+sz*(j-1)*stb, β,
        c+sz*(j-1)*stc)

B.l+B.u+1


C

A
(j-B.u-1)
(j-1)*stb

C.l+C.u+1


B

if last(cols)==m
    return C
end

# multiply columns where A and B are mid and C is bottom
cols=max(n-C.l+1,C.u+1):min(ν-B.l,n+C.u,m)
print(cols,",")
for j=cols
    @assert ((j-1)*stb + B.l+B.u+1  ≤ length(B.data))
    @assert ((j-1)*stc + n-j+C.u+1  ≤ length(C.data))
    gbmv!('N', n-j+C.u+1,
            A.l+A.u, 0,
            α,
            a+sz*(j-B.u-1)*sta, B.l+B.u+1, sta,
            b+sz*(j-1)*stb, β,
            c+sz*(j-1)*stc)
end

if last(cols)==m
    return C
end

# multiply columns where A,  B and C are bottom
cols=max(ν-B.l+1,n-C.l+1,C.u+1):min(m,n+C.u,B.u+ν)
println(cols)
for j=cols
    @assert ((j-1)*stb + B.l+B.u+1-(j-ν+B.l) ≤ length(B.data))
    @assert ((j-1)*stc + n-j+C.u+1  ≤ length(C.data))
    gbmv!('N', n-j+C.u+1,
            A.l+A.u, 0,
            α,
            a+sz*(j-B.u-1)*sta, B.l+B.u+1-(j-ν+B.l), sta,
            b+sz*(j-1)*stb, β,
            c+sz*(j-1)*stc)
end

if last(cols)==m
    return C
end

C

A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)


@time for k=1:10
    A*A
end

@time for k=1:10
    B*B
end


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



## BigFloat


B=bzeros(BigFloat,5,5,2,3)
    for (k,j)=BandedMatrices.eachbandedindex(B)
        B[k,j]=randn()
    end

x=BigFloat[1:size(B,1)...]





@test_approx_eq full(B)*x B*x
@test_approx_eq full(B*B) full(B)*full(B)
