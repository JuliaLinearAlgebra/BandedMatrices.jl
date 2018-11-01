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
function lu(A::BandedMatrix{T}) where {T<:Number}
    # copy to a blas type that allows calculation of the factorisation in place.
    S = _promote_to_blas_type(T, T)
    # copy into larger array of size (2l+u*1)×n, i.e. l additional rows
    m, n = size(A)
    data = Array{S}(undef, 2*A.l+A.u+1, n)
    data[(A.l+1):end, :] = A.data
    data, ipiv = gbtrf!(m, A.l, A.u, data)
    BandedLU{S}(data, ipiv, A.l, A.u, m)
end
lu(F::BandedLU) = F # no op

lu(A::AbstractBandedMatrix) = lu(convert(BandedMatrix, A))
lu(A::Transpose{<:Any,<:AbstractBandedMatrix}) = lu(convert(BandedMatrix, A))
lu(A::Adjoint{<:Any,<:AbstractBandedMatrix}) = lu(convert(BandedMatrix, A))

adjoint(F::BandedLU) = Adjoint(F)
transpose(F::BandedLU) = Transpose(F)

# TODO: Finish. There was something weird about the pivoting
# function Base.getproperty(F::BandedLU{T}, d::Symbol) where T
#     if d == :L
#         m, n = size(F)
#         l,u = F.l, F.u
#         return _BandedMatrix([Ones{T}(1,n); view(F.data,l+u+2:2l+u+1, :)], m, l,0)
#     elseif d == :U
#         m, n = size(F)
#         l,u = F.l, F.u
#         return _BandedMatrix(F.data[l+1:l+u+1, :], n, 0, u)
#     elseif d == :p
#         return LinearAlgebra.ipiv2perm(getfield(F, :ipiv), m)
#     elseif d == :P
#         m, n = size(F)
#         return Matrix{T}(I, m, m)[:,invperm(F.p)]
#     else
#         getfield(F, d)
#     end
# end
#
# # iteration for destructuring into components
# Base.iterate(S::BandedLU) = (S.L, Val(:U))
# Base.iterate(S::BandedLU, ::Val{:U}) = (S.U, Val(:p))
# Base.iterate(S::BandedLU, ::Val{:p}) = (S.p, Val(:done))
# Base.iterate(S::BandedLU, ::Val{:done}) = nothing

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
for typ in (BandedMatrix, BandedLU, (Transpose{V,BandedLU{V}} where V), (Adjoint{V,BandedLU{V}} where V))
    @eval function _convert_to_blas_type(A::$typ, B::AbstractVecOrMat{T}) where {T<:Number}
        TS = _promote_to_blas_type(eltype(A), eltype(B))
        AA = convert($typ{TS}, A)
        BB = convert(Array{TS, ndims(B)}, B)
        AA, BB # one of these two might make a copy
    end
end


@lazyldiv BandedMatrix
# @lazyldiv BandedLU
