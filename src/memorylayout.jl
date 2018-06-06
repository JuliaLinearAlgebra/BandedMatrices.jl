####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedLayout, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

abstract type MemoryLayout end
struct UnknownLayout <: MemoryLayout end
abstract type AbstractStridedLayout <: MemoryLayout end
abstract type AbstractColumnMajor <: AbstractStridedLayout end
struct DenseColumnMajor <: AbstractColumnMajor end
struct ColumnMajor <: AbstractColumnMajor end
abstract type AbstractRowMajor <: AbstractStridedLayout end
struct DenseRowMajor <: AbstractRowMajor end
struct RowMajor <: AbstractRowMajor end
struct StridedLayout <: AbstractStridedLayout end


struct BandedLayout <: MemoryLayout end
struct SymBandedLayout <: MemoryLayout end

MemoryLayout(A::AbstractArray) = UnknownLayout()
MemoryLayout(A::DenseVector) = DenseColumnMajor()
MemoryLayout(A::DenseMatrix) = DenseColumnMajor()

import Base: AbstractCartesianIndex, Slice, RangeIndex

MemoryLayout(A::SubArray) = submemorylayout(MemoryLayout(parent(A)), parentindices(A))
submemorylayout(::MemoryLayout, _) = UnknownLayout()
submemorylayout(::AbstractColumnMajor, ::Tuple{I}) where {I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor()
submemorylayout(::AbstractStridedLayout, ::Tuple{I}) where {I<:Union{RangeIndex,AbstractCartesianIndex}} =
    StridedLayout()
submemorylayout(::AbstractColumnMajor, ::Tuple{I,Int}) where {I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor()
submemorylayout(::AbstractColumnMajor, ::Tuple{I,Int}) where {I<:Slice} =
    DenseColumnMajor()
submemorylayout(::AbstractRowMajor, ::Tuple{Int,I}) where {I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor()
submemorylayout(::AbstractRowMajor, ::Tuple{Int,I}) where {I<:Slice} =
    DenseColumnMajor()
submemorylayout(::DenseColumnMajor, ::Tuple{I1,I2}) where {I1<:Slice,I2<:AbstractUnitRange{Int}} =
    DenseColumnMajor()
submemorylayout(::DenseColumnMajor, ::Tuple{I1,I2}) where {I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    ColumnMajor()
submemorylayout(::AbstractColumnMajor, ::Tuple{I1,I2}) where {I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    ColumnMajor()
submemorylayout(::AbstractRowMajor, ::Tuple{I1,I2}) where {I1<:AbstractUnitRange{Int},I2<:Slice} =
    DenseRowMajor()
submemorylayout(::AbstractRowMajor, ::Tuple{I1,I2}) where {I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    RowMajor()
submemorylayout(::AbstractStridedLayout, ::Tuple{I1,I2}) where {I1<:Union{RangeIndex,AbstractCartesianIndex},I2<:Union{RangeIndex,AbstractCartesianIndex}} =
    StridedLayout()



mul!(y, A, x, α, β) = _mul!(y, A, x, α, β, MemoryLayout(x), MemoryLayout(A), MemoryLayout(y))
_mul!(y, A, x, α, β, blasA, blasx, blasy) = (y .= α .* A*x .+ β.*y)
_mul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
                   ::AbstractStridedLayout, ::BandedLayout, ::AbstractStridedLayout) where {T<:BlasFloat} =
    gbmv!('N', α, A, x, β, y)
_mul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
                   ::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractStridedLayout) where {T<:BlasFloat} =
    BLAS.gemv!('N', α, A, x, β, y)

_mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, α, β,
                   ::BandedLayout, ::BandedLayout, ::BandedLayout) where {T<:BlasFloat} =
    gbmm!('N', 'N', α, A, B, β, C)
_mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, α, β,
                   ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where {T<:BlasFloat} =
    BLAS.gemm!('N', 'N', α, A, B, β, C)
