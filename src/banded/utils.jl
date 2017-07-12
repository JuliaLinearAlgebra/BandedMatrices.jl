# check if matrix is square before solution of linear system or before converting

checksquare(A::BandedMatrix) = (size(A, 1) == size(A, 2) ||
    throw(ArgumentError("Banded matrix must be square")))

checksquare(A::BandedLU) = (A.m == size(A.data, 2) ||
    throw(ArgumentError("Banded matrix must be square")))


# return the bandwidths of A*B
function prodbandwidths(A::AbstractMatrix, B::AbstractMatrix)
    m = size(A, 1)
    n = size(B, 2)
    min(bandwidth(A, 1) + bandwidth(B, 1), m-1), min(bandwidth(A, 2) + bandwidth(B, 2), n-1)
end

# return the bandwidths of A+B
function sumbandwidths(A::AbstractMatrix, B::AbstractMatrix)
    max(bandwidth(A, 1), bandwidth(B, 1)), max(bandwidth(A, 2), bandwidth(B, 2))
end
