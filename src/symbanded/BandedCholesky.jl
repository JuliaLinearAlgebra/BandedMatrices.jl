## Banded Cholesky decomposition

struct BandedCholesky{T} <: Factorization{T}
    data::SymBandedMatrix{T}  # symmetric banded matrix
end

# conversion
convert(::Type{BandedCholesky{T}}, B::BandedCholesky{S}) where {T<:Number, S<:Number} =
    BandedCholesky{T}(convert(SymBandedMatrix{T}, B.data))

# size of the parent array
@inline size(A::BandedCholesky) = size(A.data)
@inline size(A::BandedCholesky, k::Integer) = size(A.data, k)
@inline bandwidth(A::BandedCholesky, k) = bandwidth(A.data, k)

# Cholesky factorisation.
function cholfact!(A::SymBandedMatrix{T}) where {T<:Number}
    pbtrf!('U', size(A, 1), bandwidth(A, 2), pointer(A), leadingdimension(A))
    BandedCholesky{T}(A)
end
cholfact(A::SymBandedMatrix) = cholfact!(copy(A))
cholfact(F::BandedCholesky) = F # no op

function getindex(F::BandedCholesky, d::Symbol)
    d == :U && return BandedMatrix(F.data.data, size(F, 1), 0, bandwidth(F, 2)) # UpperTriangular(F.data)
    d == :L && return BandedMatrix(F.data.data, size(F, 1), 0, bandwidth(F, 2))' # LowerTriangular(F.data)
    throw(KeyError(d))
end

## Utilities

# check if matrix is square before solution of linear system or before converting
checksquare(A::BandedCholesky) = checksquare(A.data)


## Conversion/Promotion

# Converts A and b to the narrowest blas type
for typ in [BandedCholesky]
    @eval function _convert_to_blas_type(A::$typ, B::AbstractVecOrMat{T}) where {T<:Number}
        TS = _promote_to_blas_type(eltype(A), eltype(B))
        AA = convert($typ{TS}, A)
        BB = convert(Array{TS, ndims(B)}, B)
        AA, BB # one of these two might make a copy
    end
end
