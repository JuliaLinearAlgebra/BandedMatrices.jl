using BandedMatrices, LazyArrays, Test
    import BandedMatrices: MemoryLayout



A = UpperTriangular(brand(10,10,1,2))
MemoryLayout(A)


y = randn(10)
@which MemoryLayout(Inv(A))

Mul(Inv(A), y)


@test isbanded(A)
@test BandedMatrix(A) == A
@test bandwidths(A) == bandwidths(BandedMatrix(A)) == (0,2)
@test MemoryLayout(A) == SymmetricLayout(BandedColumnMajor(), 'U')



@test A[1,2] == A[2,1]
@test A[1,4] == 0
x=rand(10)
@test A*x ≈ Matrix(A)*x
@test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= 1.0.*Mul(A,x) .+ 0.0.*similar(x)) .===
            BLAS.sbmv!('U', 2, 1.0, parent(A).data, x, 0.0, similar(x)))


A = Symmetric(brand(10,10,1,2),:L)
@test isbanded(A)
@test BandedMatrix(A) == A
@test bandwidth(A) == bandwidth(A,1) == bandwidth(A,2) ==  1
@test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,1)
@test MemoryLayout(A) == SymmetricLayout(BandedColumnMajor(), 'L')

@test A[1,2] == A[2,1]
@test A[1,3] == 0
x=rand(10)
@test A*x ≈ Matrix(A)*x

@test all(A*x .=== (similar(x) .= Mul(A,x)) .=== (similar(x) .= 1.0.*Mul(A,x) .+ 0.0.*similar(x)) .===
            BLAS.sbmv!('L', 1, 1.0, view(parent(A).data,3:4,:), x, 0.0, similar(x)))

# eigvals
Random.seed!(0)

A = Symmetric(brand(Float64, 100, 100, 0, 4))
@test eigvals(A) ≈ eigvals(Symmetric(Matrix(A)))


# generalized eigvals

function An(::Type{T}, N::Int) where {T}
    A = Symmetric(BandedMatrix(Zeros{T}(N,N), (0, 2)))
    for n = 0:N-1
        parent(A).data[3,n+1] = T((n+1)*(n+2))
    end
    A
end

function Bn(::Type{T}, N::Int) where {T}
    B = Symmetric(BandedMatrix(Zeros{T}(N,N), (0,2)))
    for n = 0:N-1
        parent(B).data[3,n+1] = T(2*(n+1)*(n+2))/T((2n+1)*(2n+5))
    end
    for n = 0:N-3
        parent(B).data[1,n+3] = -sqrt(T((n+1)*(n+2)*(n+3)*(n+4))/T((2n+3)*(2n+5)*(2n+5)*(2n+7)))
    end
    B
end

A = An(Float64, 100)
B = Bn(Float64, 100)

λ = eigvals(A, B)
@test λ ≈ eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(B)))

err = λ*(2/π)^2 ./ (1:length(λ)).^2 .- 1

@test norm(err[1:40]) < 100eps(Float64)

A = An(Float32, 100)
B = Bn(Float32, 100)

λ = eigvals(A, B)

@test λ ≈ eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(B)))

err = λ*(2.f0/π)^2 ./ (1:length(λ)).^2 .- 1

@test norm(err[1:40]) < 100eps(Float32)
