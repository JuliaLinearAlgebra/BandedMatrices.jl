################################
# split-Cholesky Factorization #
################################

#
# A split-Cholesky factorization of an SPD banded matrix A is:
#
#   S = [ U    ]
#       [ M  L ],
#
# where U and M are upper-triangular and L is lower-triangular so that S' * S = A.
# For a real A, the matrix S is also real, and S' == transpose(S)
#
# S has the same bandwidth as A
#
# If n = size(A, 1), n == size(A, 2) and kd = bandwidth(A), then if m = (n+b)÷2,
#
# size(U) == (m, m),  size(L) == (n-m, n-m),  and size(M) == (n-m, m).
#

struct SplitCholesky{T,M<:AbstractMatrix{T}} <: Factorization{T}
    factors::M
    uplo::Char

    function SplitCholesky{T,M}(factors, uplo) where {T,M<:AbstractMatrix}
        require_one_based_indexing(factors)
        new{T,M}(factors, uplo)
    end
end

SplitCholesky(factors::AbstractMatrix{T}, uplo::Char) where T = SplitCholesky{T,typeof(factors)}(factors, uplo)

size(S::SplitCholesky) = size(S.factors)
size(S::SplitCholesky, i::Integer) = size(S.factors, i)

function splitcholesky!(A::HermOrSym{T,<:BandedMatrix{T}}) where T
    splitcholesky!(A, A.uplo == 'U' ? UpperTriangular : LowerTriangular)
end
function splitcholesky!(A::HermOrSym{T,<:BandedMatrix{T}}, Tr) where {T}
    splitcholesky!(MemoryLayout(typeof(A)), A, Tr)
end

function splitcholesky!(::SymmetricLayout{<:BandedColumnMajor},
                       A::AbstractMatrix{T}, ::Type{LU}) where {T<:BlasFloat, LU}
    uplo = LU == UpperTriangular ? 'U' : 'L'
    pbstf!(uplo, size(A, 1), bandwidth(A,2), symbandeddata(A))
    SplitCholesky(A, A.uplo)
end

function splitcholesky!(::HermitianLayout{<:BandedColumnMajor},
                       A::AbstractMatrix{T}, ::Type{LU}) where {T<:BlasFloat, LU}
    uplo = LU == UpperTriangular ? 'U' : 'L'
    pbstf!(uplo, size(A, 1), bandwidth(A,2), hermbandeddata(A))
    SplitCholesky(A, A.uplo)
end

if !(isdefined(LinearAlgebra, :AdjointFactorization)) # VERSION < v"1.10-"
    adjoint(S::SplitCholesky) = Adjoint(S)
else
    transpose(S::SplitCholesky{<:Real}) = S'
    transpose(::SplitCholesky) =
        throw(ArgumentError("transpose of SplitCholesky decomposition is not supported, consider using adjoint"))
end

function lmul!(S::SplitCholesky{T,<:HermOrSym{T,M}}, B::AbstractVecOrMat{T}) where {T<:Real,M<:BandedMatrix{T}}
    require_one_based_indexing(B)
    n, nrhs = size(B, 1), size(B, 2)
    if size(S, 1) != n
        throw(DimensionMismatch("Matrix has dimensions $(size(S)) but right hand side has first dimension $n"))
    end
    A = S.factors
    b = bandwidth(A, 1)
    m = (n+b)÷2
    @inbounds for l = 1:nrhs
        for j = n:-1:m+1
            t = zero(T)
            @simd for k = j-b:j
                t += A[j,k]*B[k,l]
            end
            B[j,l] = t
        end
        for j = 1:m
            t = zero(T)
            @simd for k = j:min(j+b,m)
                t += A[j,k]*B[k,l]
            end
            B[j,l] = t
        end
    end
    B
end

function ldiv!(S::SplitCholesky{T,<:HermOrSym{T,M}}, B::AbstractVecOrMat{T}) where {T<:Real,M<:BandedMatrix{T}}
    require_one_based_indexing(B)
    n, nrhs = size(B, 1), size(B, 2)
    if size(S, 1) != n
        throw(DimensionMismatch("Matrix has dimensions $(size(S)) but right hand side has first dimension $n"))
    end
    A = S.factors
    b = bandwidth(A, 1)
    m = (n+b)÷2
    @inbounds for l = 1:nrhs
        for j = m:-1:1
            t = zero(T)
            @simd for k = j+1:min(j+b,m)
                t += A[j,k]*B[k,l]
            end
            B[j,l] = (B[j,l]-t)/A[j,j]
        end
        for j = m+1:n
            t = zero(T)
            @simd for k = j-b:j-1
                t += A[j,k]*B[k,l]
            end
            B[j,l] = (B[j,l]-t)/A[j,j]
        end
    end
    B
end

function lmul!(S::AdjointFact{T,<:SplitCholesky{T,<:HermOrSym{T,M}}}, B::AbstractVecOrMat{T}) where {T<:Real,M<:BandedMatrix{T}}
    require_one_based_indexing(B)
    n, nrhs = size(B, 1), size(B, 2)
    if size(S, 1) != n
        throw(DimensionMismatch("Matrix has dimensions $(size(S)) but right hand side has first dimension $n"))
    end
    A = S.parent.factors
    b = bandwidth(A, 1)
    m = (n+b)÷2
    @inbounds for l = 1:nrhs
        for j = m:-1:1
            t = zero(T)
            @simd for k = max(1,j-b):j
                t += A[k,j]*B[k,l]
            end
            B[j,l] = t
        end
        for j = m-b+1:m
            t = zero(T)
            @simd for k = m+1:j+b
                t += A[k,j]*B[k,l]
            end
            B[j,l] += t
        end
        for j = m+1:n
            t = zero(T)
            @simd for k = j:min(j+b,n)
                t += A[k,j]*B[k,l]
            end
            B[j,l] = t
        end
    end
    B
end

function ldiv!(S::AdjointFact{T,<:SplitCholesky{T,<:HermOrSym{T,M}}}, B::AbstractVecOrMat{T}) where {T<:Real,M<:BandedMatrix{T}}
    require_one_based_indexing(B)
    n, nrhs = size(B, 1), size(B, 2)
    if size(S, 1) != n
        throw(DimensionMismatch("Matrix has dimensions $(size(S)) but right hand side has first dimension $n"))
    end
    A = S.parent.factors
    b = bandwidth(A, 1)
    m = (n+b)÷2
    @inbounds for l = 1:nrhs
        for j = n:-1:m+1
            t = zero(T)
            @simd for k = j+1:min(j+b,n)
                t += A[k,j]*B[k,l]
            end
            B[j,l] = (B[j,l]-t)/A[j,j]
        end
        for j = m:-1:m-b+1
            t = zero(T)
            @simd for k = m+1:j+b
                t += A[k,j]*B[k,l]
            end
            B[j,l] -= t
        end
        for j = 1:m
            t = zero(T)
            @simd for k = max(1,j-b):j-1
                t += A[k,j]*B[k,l]
            end
            B[j,l] = (B[j,l]-t)/A[j,j]
        end
    end
    B
end
