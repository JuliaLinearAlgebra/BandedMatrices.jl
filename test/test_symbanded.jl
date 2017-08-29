using BandedMatrices, Base.Test


A = sbrand(10,2)

@test A[1,2] == A[2,1]


b=rand(10)
@test A*b ≈ full(A)*b


# eig and eigvals

srand(0)
for T in (Float32, Float64)
    A = sbrand(T, 100, 4)
    @test eigvals(A) ≈ eigvals(Symmetric(full(A)))

    Λ, Q = eig(A)
    Λf, Qf = eig(Symmetric(full(A)))
    @test Λ ≈ Λf
    @test A ≈ Q*Diagonal(Λ)*Q'

    B = sbrand(T, 100, 2)
    [B[i,i] += 2 for i = 1:100] # make it positive definite
    @test eigvals(A, B) ≈ eigvals(Symmetric(full(A)), Symmetric(full(B)))

    Λ, Q = eig(A, B)
    Λf, Qf = eig(Symmetric(full(A)), Symmetric(full(B)))
    @test Λ ≈ Λf

    @test Q'A*Q ≈ Diagonal(Λ)
    @test Q'B*Q ≈ eye(T, 100)
end

# generalized eigvals

function An(::Type{T}, N::Int) where {T}
    A = sbzeros(T, N, 2)
    for n = 0:N-1
        A.data[3,n+1] = T((n+1)*(n+2))
    end
    A
end

function Bn(::Type{T}, N::Int) where {T}
    B = sbzeros(T, N, 2)
    for n = 0:N-1
        B.data[3,n+1] = T(2*(n+1)*(n+2))/T((2n+1)*(2n+5))
    end
    for n = 0:N-3
        B.data[1,n+3] = -sqrt(T((n+1)*(n+2)*(n+3)*(n+4))/T((2n+3)*(2n+5)*(2n+5)*(2n+7)))
    end
    B
end

A = An(Float64, 100)
B = Bn(Float64, 100)

λ = eigvals(A, B)

@test λ ≈ eigvals(Symmetric(full(A)), Symmetric(full(B)))

err = λ*(2/π)^2 ./ (1:length(λ)).^2-1

@test norm(err[1:40]) < 100eps(Float64)

A = An(Float32, 100)
B = Bn(Float32, 100)

λ = eigvals(A, B)

@test λ ≈ eigvals(Symmetric(full(A)), Symmetric(full(B)))

err = λ*(2.f0/π)^2 ./ (1:length(λ)).^2-1

@test norm(err[1:40]) < 100eps(Float32)
