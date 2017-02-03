using BandedMatrices, Base.Test


A = sbrand(10,2)

@test A[1,2] == A[2,1]


b=rand(10)
@test A*b ≈ full(A)*b


# eigvals

srand(0)
A = sbrand(Float64,100,4)
@test eigvals(A) ≈ eigvals(full(A))
