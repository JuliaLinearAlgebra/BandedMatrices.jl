# SubArray

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
