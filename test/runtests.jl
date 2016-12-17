versioninfo()

using BandedMatrices, Compat, Base.Test
    import Compat: view

include("test_indexing.jl")
include("test_bandedlu.jl")
include("test_bandedqr.jl")

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

for j = 1:size(A,2), k = colrange(A,j)
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

# test gbmm! subpieces step by step and column by column
for n in (1,5,50), ν in (1,5,50), m in (1,5,50),
                Al in (0,1,2,30), Au in (0,1,2,30),
                Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    α,β,T=0.123,0.456,Float64
    C=brand(Float64,n,m,A.l+B.l,A.u+B.u)
    a=pointer(A.data)
    b=pointer(B.data)
    c=pointer(C.data)
    sta=max(1,stride(A.data,2))
    stb=max(1,stride(B.data,2))
    stc=max(1,stride(C.data,2))

    sz=sizeof(T)

    mr=1:min(m,1+B.u)
    exC=(β*full(C)+α*full(A)*full(B))
    for j=mr
        BandedMatrices.A11_Btop_Ctop_gbmv!(α,β,
                                       n,ν,m,j,
                                       sz,
                                       a,A.l,A.u,sta,
                                       b,B.l,B.u,stb,
                                       c,C.l,C.u,stc)
   end
    @test_approx_eq C[:,mr] exC[:,mr]

    mr=1+B.u:min(1+C.u,ν+B.u,m)
    exC=(β*full(C)+α*full(A)*full(B))
    for j=mr
        BandedMatrices.Atop_Bmid_Ctop_gbmv!(α,β,
                                       n,ν,m,j,
                                       sz,
                                       a,A.l,A.u,sta,
                                       b,B.l,B.u,stb,
                                       c,C.l,C.u,stc)
   end
   if !isempty(mr)
       @test_approx_eq C[:,mr] exC[:,mr]
   end

   mr=1+C.u:min(m,ν+B.u,n+C.u)
   exC=(β*full(C)+α*full(A)*full(B))
   for j=mr
       BandedMatrices.Amid_Bmid_Cmid_gbmv!(α,β,
                                      n,ν,m,j,
                                      sz,
                                      a,A.l,A.u,sta,
                                      b,B.l,B.u,stb,
                                      c,C.l,C.u,stc)
  end
  if !isempty(mr)
      @test_approx_eq C[:,mr] exC[:,mr]
  end

  mr=ν+B.u+1:min(m,n+C.u)
  exC=(β*full(C)+α*full(A)*full(B))
  for j=mr
      BandedMatrices.Anon_Bnon_C_gbmv!(α,β,
                                     n,ν,m,j,
                                     sz,
                                     a,A.l,A.u,sta,
                                     b,B.l,B.u,stb,
                                     c,C.l,C.u,stc)
 end
 if !isempty(mr)
     @test_approx_eq C[:,mr] exC[:,mr]
 end
end


# test gbmm!


for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
    A=brand(n,ν,Al,Au)
    B=brand(ν,m,Bl,Bu)
    α,β,T=0.123,0.456,Float64
    C=brand(Float64,n,m,A.l+B.l,A.u+B.u)

    exC=α*full(A)*full(B)+β*full(C)
    BandedMatrices.gbmm!(α,A,B,β,C)

    @test_approx_eq full(exC) full(C)
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

@time for k=1:10
    B*B
end


#gc_enable(false)

A*v
@time for k=1:100
    A*v
end
println("Time should be   0.004986 seconds (400 allocations: 7.639 MB)")
@time for k=1:100
    B*w
end
println("Time should be   0.017208 seconds (300 allocations: 792.188 KB)")


A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)


@time for k=1:10
    A*A
end

println("Time should be   0.008115 seconds (30 allocations: 8.393 MB)")

@time for k=1:10
    B*B
end

println("Time should be   0.644119 seconds (30 allocations: 76.371 MB)")

#gc_enable(true)




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




## BigFloat


B=bzeros(BigFloat,5,5,2,3)
    for j = 1:size(B,2), k = colrange(B,j)
        B[k,j]=randn()
    end

x=BigFloat[1:size(B,1)...]





@test_approx_eq full(B)*x B*x
@test_approx_eq full(B*B) full(B)*full(B)



A = brand(10,10,1,2)
B = brand(20,20,1,2)

@test isa(view(B,1:10,1:10),BandedMatrices.BandedSubMatrix{Float64})

B2 = copy(B)
@test_approx_eq (2.0A+B[1:10,1:10]) BLAS.axpy!(2.0,A,view(B2,1:10,1:10))
@test_approx_eq (2.0A+B[1:10,1:10]) B2[1:10,1:10]

A2 = copy(A)
@test_approx_eq (2.0B[1:10,1:10]+A) BLAS.axpy!(2.0,view(B,1:10,1:10),A2)
@test_approx_eq (2.0B[1:10,1:10]+A) A2

A = brand(20,20,1,2)
B = brand(20,20,1,2)

@test isa(view(B,1:10,1:10),BandedMatrices.BandedSubMatrix{Float64})

B2 = copy(B)
@test_approx_eq (2.0A[1:10,1:10]+B[1:10,1:10]) BLAS.axpy!(2.0,view(A,1:10,1:10),view(B2,1:10,1:10))
@test_approx_eq (2.0A[1:10,1:10]+B[1:10,1:10]) B2[1:10,1:10]

B2 = copy(B)
@test_approx_eq (2.0A[1:10,:]+B[1:10,:]) BLAS.axpy!(2.0,view(A,1:10,:),view(B2,1:10,:))
@test_approx_eq (2.0A[1:10,:]+B[1:10,:]) B2[1:10,:]

B2 = copy(B)
@test_approx_eq (2.0A[:,1:10]+B[:,1:10]) BLAS.axpy!(2.0,view(A,:,1:10),view(B2,:,1:10))
@test_approx_eq (2.0A[:,1:10]+B[:,1:10]) B2[:,1:10]


B2 = copy(B)
@test_approx_eq (2.0A[:,:]+B[:,:]) BLAS.axpy!(2.0,view(A,:,:),view(B2,:,:))
@test_approx_eq (2.0A[:,:]+B[:,:]) B2[:,:]



A = brand(10,10,1,2)
B = brand(20,10,1,2)

B2 = copy(B)
@test_approx_eq (2.0A+B[1:10,1:10]) BLAS.axpy!(2.0,A,view(B2,1:10,:))
@test_approx_eq (2.0A+B[1:10,1:10]) B2[1:10,1:10]

A2 = copy(A)
@test_approx_eq (2.0B[1:10,1:10]+A) BLAS.axpy!(2.0,view(B,1:10,:),A2)
@test_approx_eq (2.0B[1:10,1:10]+A) A2


A = brand(10,10,1,2)
B = brand(10,20,1,2)

B2 = copy(B)
@test_approx_eq (2.0A+B[1:10,1:10]) BLAS.axpy!(2.0,A,view(B2,:,1:10))
@test_approx_eq (2.0A+B[1:10,1:10]) B2[1:10,1:10]

A2 = copy(A)
@test_approx_eq (2.0B[1:10,1:10]+A) BLAS.axpy!(2.0,view(B,:,1:10),A2)
@test_approx_eq (2.0B[1:10,1:10]+A) A2



## UniformScalin g
A = brand(10,10,1,2)

@test_approx_eq full(A+I) full(A)+I
@test_approx_eq full(I-A) I-full(A)


@test_approx_eq full(A/2) full(A)/2


## Test StridedMatrix
A=brand(10,10,1,2)
v=rand(20)

@test_approx_eq A*view(v,1:10) full(A)*v[1:10]
@test_approx_eq A*view(v,1:2:20) full(A)*v[1:2:20]

M=rand(20,20)
@test_approx_eq A*view(M,1:10,1:10) full(A)*M[1:10,1:10]
@test_approx_eq A*view(M,1:2:20,1:2:20) full(A)*M[1:2:20,1:2:20]


A=brand(10,10,1,2)
B=brand(20,10,1,2)
C=brand(10,20,1,2)
D=brand(20,20,1,2)
M=rand(10,10)
V=view(rand(20,20),1:2:20,1:2:20)


for S in (view(A,:,:),view(B,1:10,:),view(C,:,1:10),view(D,1:10,1:10),
            view(D,2:11,1:10),view(D,1:10,2:11),view(D,11:20,11:20))
    @test_approx_eq A*S full(A)*full(S)
    @test_approx_eq S*A full(S)*full(A)
    @test_approx_eq S*S full(S)*full(S)

    @test_approx_eq M*S M*full(S)
    @test_approx_eq S*M full(S)*M

    @test_approx_eq V*S full(V)*full(S)
    @test_approx_eq S*V full(S)*full(V)
end


# negative bands

for A in (brand(3,4,-1,2),brand(5,4,-1,2),
            brand(3,4,2,-1),brand(5,4,2,-1))
    b = rand(size(A,2))
    @test_approx_eq A*b full(A)*b
end


for A in (brand(3,4,1,2),brand(3,4,-1,2),brand(3,4,2,-1)),
    B in (brand(4,4,1,2),brand(4,4,-1,2),brand(4,4,2,-1))
    @test_approx_eq A*B full(A)*full(B)
end



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


include("test_symbanded.jl")
