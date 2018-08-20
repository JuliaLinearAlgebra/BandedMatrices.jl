####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedLayout, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

struct BandedColumnMajor <: MemoryLayout end
struct BandedRowMajor <: MemoryLayout end

# make copy to make sure always works
@inline function _gbmv!(tA, α, A, x, β, y)
    if x ≡ y
        BLAS.gbmv!(tA, α, A, copy(x), β, y)
    else
        BLAS.gbmv!(tA, α, A, x, β, y)
    end
end

@blasmatvec BandedColumnMajor

@inline blasmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α, β,
              ::AbstractStridedLayout, ::BandedColumnMajor, ::AbstractStridedLayout) =
    _gbmv!('N', α, A, x, β, y)

@blasmatvec BandedRowMajor

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::BandedRowMajor, ::AbstractStridedLayout) where T<:BlasReal =
    _gbmv!('T', α, transpose(A), x, β, y)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::BandedRowMajor, ::AbstractStridedLayout) where T<:BlasComplex =
    _gbmv!('C', α, A', x, β, y)
