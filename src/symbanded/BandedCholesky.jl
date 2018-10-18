## Banded Cholesky decomposition

struct BandedCholesky{T} <: Factorization{T}
    data::BandedMatrix{T}  # symmetric banded matrix
end

# conversion
convert(::Type{BandedCholesky{T}}, B::BandedCholesky{S}) where {T<:Number, S<:Number} =
    BandedCholesky{T}(convert(BandedMatrix{T}, B.data))

# size of the parent array
@inline size(A::BandedCholesky) = size(A.data)
@inline size(A::BandedCholesky, k::Integer) = size(A.data, k)
@inline bandwidth(A::BandedCholesky, k) = bandwidth(A.data, k)

# Cholesky factorisation.
function cholesky!(A::Union{Symmetric{R,<:BandedMatrix},Hermitian{Complex{R},<:BandedMatrix}}) where {R<:Real}
    P = parent(A)
    pbtrf!('U', size(A, 1), bandwidth(A), bandeddata(parent(P)))
    BandedCholesky{eltype(A)}(P)
end
cholesky(A::Union{Symmetric{R,<:BandedMatrix},Hermitian{Complex{R},<:BandedMatrix}}) where {R<:Real} = cholesky!(copy(A))
cholesky(F::BandedCholesky) = F # no op

function getindex(F::BandedCholesky, d::Symbol)
    d == :U && return F.data # UpperTriangular(F.data)
    d == :L && return F.data' # LowerTriangular(F.data)
    throw(KeyError(d))
end

function ldiv!(A::BandedCholesky{T}, B::AbstractVecOrMat{T}) where T
    checksquare(A)
    m = size(A,1)
    pbtrs!('U', size(A, 1), bandwidth(A,2), A.data.data, B)
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
