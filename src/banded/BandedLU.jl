## Banded LU decomposition

# this is currently a hack and not the nicest of the interfaces.
# if the LAPACK storage was built in the BandedMatrix we could get rid of
# the last three fields of this type, and have the first point to the parent
# BandedMatrix, rather then to a generic chunk of memory
struct BandedLU{T} <: Factorization{T}
    data::Matrix{T}        # banded matrix plus additional storage
    ipiv::Vector{BlasInt}  # vector of pivots
    l::Int                 # lower bandwidth
    u::Int                 # upper bandwidth
    m::Int                 # number of rows
end

# conversion
convert(::Type{BandedLU{T}}, B::BandedLU{S}) where {T<:Number, S<:Number} =
    BandedLU{T}(convert(Matrix{T}, B.data), B.ipiv, B.l, B.u, B.m)

# size of the parent array
size(A::BandedLU) = (A.m, size(A.data, 2))
size(A::BandedLU, k::Integer) = k <= 0 ? error("dimension out of range") :
                                k == 1 ? A.m :
                                k == 2 ? size(A.data, 2) : 1

# LU factorisation with pivoting. This makes a copy!
function lufact(A::BandedMatrix{T}) where {T<:Number}
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


checksquare(A::BandedLU) = (A.m == size(A.data, 2) ||
    throw(DimensionMismatch("matrix must be matrix is not square: dimensions are $(size(A))")))

## Conversion/Promotion

# Returns the narrowest blas type given the eltypes of A and b in A*x=b
function _promote_to_blas_type(::Type{T}, ::Type{S}) where {T<:Number, S<:Number}
    TS = Base.promote_op(/, T, S)
    # promote to narrowest type
    TS <: Complex       && return Base.promote_op(/, TS, ComplexF32)
    TS <: AbstractFloat && return Base.promote_op(/, TS, Float32)
    error("Cannot convert objects of element type $(T), $(S) to a `BlasFloat` type")
end

# Converts A and b to the narrowest blas type
for typ in [BandedMatrix, BandedLU]
    @eval function _convert_to_blas_type(A::$typ, B::AbstractVecOrMat{T}) where {T<:Number}
        TS = _promote_to_blas_type(eltype(A), eltype(B))
        AA = convert($typ{TS}, A)
        BB = convert(Array{TS, ndims(B)}, B)
        AA, BB # one of these two might make a copy
    end
end


# basic interface
(\)(A::Union{BandedLU{T}, BandedMatrix{T}}, B::StridedVecOrMat{T}) where {T<:BlasFloat} =
    A_ldiv_B!(A, copy(B)) # makes a copy
