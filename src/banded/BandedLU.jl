## Banded LU decomposition

# this is currently a hack and not the nicest of the interfaces.
# if the LAPACK storage was built in the BandedMatrix we could get rid of
# the last three fields of this type, and have the first point to the parent
# BandedMatrix, rather then to a generic chunk of memory
immutable BandedLU{T} <: Factorization{T}
    data::Matrix{T}        # banded matrix plus additional storage
    ipiv::Vector{BlasInt}  # vector of pivots
    l::Int                 # lower bandwidth
    u::Int                 # upper bandwidth
    m::Int                 # number of rows
end

# conversion
convert{T<:Number, S<:Number}(::Type{BandedLU{T}}, B::BandedLU{S}) =
    BandedLU{T}(convert(Matrix{T}, B.data), B.ipiv, B.l, B.u, B.m)

# size of the parent array
size(A::BandedLU) = (A.m, size(A.data, 2))
size(A::BandedLU, k::Integer) = k <= 0 ? error("dimension out of range") :
                                k == 1 ? A.m :
                                k == 2 ? size(A.data, 2) : 1

# LU factorisation with pivoting. This makes a copy!
function lufact{T<:Number}(A::BandedMatrix{T})
    # copy to a blas type that allows calculation of the factorisation in place.
    S = _promote_to_blas_type(T, T)
    # copy into larger array of size (2l+u*1)×n, i.e. l additional rows
    m, n = size(A)
    data = Array{S}(2*A.l+A.u+1, n)
    data[(A.l+1):end, :] = A.data
    data, ipiv = gbtrf!(A.l, A.u, m, data)
    BandedLU{S}(data, ipiv, A.l, A.u, m)
end
lufact(F::BandedLU) = F # no op


## Utilities

# check if matrix is square before solution of linear system or before converting
checksquare(A::BandedMatrix) = (size(A, 1) == size(A, 2) ||
    throw(ArgumentError("Banded matrix must be square")))
checksquare(A::BandedLU) = (A.m == size(A.data, 2) ||
    throw(ArgumentError("Banded matrix must be square")))


## Conversion/Promotion

# Returns the narrowest blas type given the eltypes of A and b in A*x=b
function _promote_to_blas_type{T<:Number, S<:Number}(::Type{T}, ::Type{S})
    TS = Base.promote_op(/, T, S)
    # promote to narrowest type
    TS <: Complex       && return Base.promote_op(/, TS, Complex64)
    TS <: AbstractFloat && return Base.promote_op(/, TS, Float32)
    error("Cannot convert objects of element type $(T), $(S) to a `BlasFloat` type")
end

# Converts A and b to the narrowest blas type
for typ in [BandedMatrix, BandedLU]
    @eval function _convert_to_blas_type{T<:Number}(A::$typ, B::AbstractVecOrMat{T})
        TS = _promote_to_blas_type(eltype(A), eltype(B))
        AA = convert($typ{TS}, A)
        BB = convert(Array{TS, ndims(B)}, B)
        AA, BB # one of these two might make a copy
    end
end


# basic interface
(\){T<:BlasFloat}(A::Union{BandedLU{T}, BandedMatrix{T}}, B::StridedVecOrMat{T}) =
    A_ldiv_B!(A, copy(B)) # makes a copy
