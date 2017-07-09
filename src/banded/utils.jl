# check if matrix is square before solution of linear system or before converting

checksquare(A::BandedMatrix) = (size(A, 1) == size(A, 2) ||
    throw(ArgumentError("Banded matrix must be square")))

checksquare(A::BandedLU) = (A.m == size(A.data, 2) ||
    throw(ArgumentError("Banded matrix must be square")))
