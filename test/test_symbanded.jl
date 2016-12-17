using BandedMatrices, Base.Test


A = sbrand(10,2)

@test A[1,2] == A[2,1]


b=rand(10)
@test_approx_eq A*b full(A)*b
