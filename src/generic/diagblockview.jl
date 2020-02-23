using BandedMatrices, BlockBandedMatrices, BlockArrays
import BandedMatrices: bandeddata
import Base: to_indices, iterate, length, getindex, view, unsafe_view, SubArray, @_inline_meta, viewindexing, ensure_indexable, index_dimsum, reindex, strides, unsafe_convert
import BlockArrays: BlockSlice


struct DiagBlock 
    block::Int
end

struct DiagBlockSlice
    block::Int
    inds::NTuple{2,UnitRange{Int}}
end

iterate(d::DiagBlockSlice, st...) = iterate(d.inds, st...)
length(d::DiagBlockSlice) = length(d.inds)

@inline function to_indices(A, inds, I::Tuple{DiagBlock})
    l,u = bandwidths(A)
    l += 1
    K = I[1].block
    DiagBlockSlice(K,((K-1)*l+1:K*l,(K-1)*l+1:K*l))
end
viewindexing(I::DiagBlockSlice) = IndexCartesian()
ensure_indexable(I::DiagBlockSlice) = I
view(A::AbstractMatrix, I::DiagBlock) = SubArray(A, to_indices(A, (I,)))
reindex(B::DiagBlockSlice, I::Tuple) = reindex(B.inds, I)
function SubArray(parent::AbstractMatrix, indices::DiagBlockSlice)
    @_inline_meta
    SubArray(IndexStyle(viewindexing(indices), IndexStyle(parent)), parent, ensure_indexable(indices), index_dimsum(indices...))
end

function strides(V::SubArray{<:Any,2,<:Any,DiagBlockSlice})
    A = parent(V)
    data = bandeddata(A)
    (stride(data,1),stride(data,2)-1)
end

function unsafe_convert(::Type{Ptr{T}}, V::SubArray{<:Any,2,<:Any,DiagBlockSlice}) where T
    A = parent(V)
    data = bandeddata(A)
    s,t = strides(data)
    K = parentindices(V)
    _,jr = K
    p = unsafe_convert(Ptr{T}, data)
    K.block == 1 && return p+s*bandwidth(A,2)
    p + t*(first(kr)-1)
end


A = brand(10,10,3,3)
V = view(A,DiagBlock(1))
BLAS.gemv('N',1.0,V,randn(4))