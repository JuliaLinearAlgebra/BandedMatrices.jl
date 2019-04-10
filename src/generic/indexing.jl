
# fallbacks for inbands_getindex and inbands_setindex!
@inline function inbands_getindex(x::AbstractArray, i::Integer, j::Integer)
    @inbounds r = getindex(x, i, j)
    r
end
@inline function inbands_setindex!(x::AbstractArray, v, i::Integer, j::Integer)
    @inbounds r = setindex!(x, v, i, j)
    r
end

inbands_getindex(x::Adjoint, i::Integer, j::Integer) =
    inbands_getindex(parent(x), j, i)'
inbands_getindex(x::Transpose, i::Integer, j::Integer) =
    transpose(inbands_getindex(parent(x), j, i))
inbands_setindex!(x::Adjoint, v, i::Integer, j::Integer) =
    inbands_setindex!(parent(x), v', j, i)
inbands_setindex!(x::Transpose, v, i::Integer, j::Integer) =
    inbands_setindex!(parent(x), transpose(v), j, i)

###
# Lazy getindex
# this uses a lazy-materialize idiom to construct a matrix based
# on the memory layout
###

@inline _sub_materialize(_, V) = Array(V)
@inline _sub_materialize(::AbstractBandedLayout, V) = BandedMatrix(V)
@inline _materialize(V::SubArray) = _sub_materialize(MemoryLayout(V), V)

@inline _lazy_getindex(A, I...) = _materialize(view(A, I...))
@inline getindex(A::AbstractBandedMatrix, kr::Colon, jr::Colon) = _lazy_getindex(A, kr, jr)
@inline getindex(A::AbstractBandedMatrix, kr::Colon, jr::AbstractUnitRange) = _lazy_getindex(A, kr, jr)
@inline getindex(A::AbstractBandedMatrix, kr::AbstractUnitRange, jr::Colon) = _lazy_getindex(A, kr, jr)
@inline getindex(A::AbstractBandedMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = _lazy_getindex(A, kr, jr)

@inline getindex(A::AbstractMatrix, b::Band) = _lazy_getindex(A, b)
@inline getindex(A::AbstractMatrix, kr::BandRangeType, j::Integer) = _lazy_getindex(A, kr, j)
@inline getindex(A::AbstractMatrix, k::Integer, jr::BandRangeType) = _lazy_getindex(A, k, jr)
