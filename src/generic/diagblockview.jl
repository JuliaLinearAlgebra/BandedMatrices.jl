using BandedMatrices, BlockBandedMatrices, BlockArrays

import LinearAlgebra.BLAS: @blasfunc, BlasInt, require_one_based_indexing, chkstride1
import LinearAlgebra.LAPACK: chklapackerror, liblapack

for (tpqrt,elty) in ((:dtpqrt2_,:Float64),)
    @eval function tpqrt2!(l_in::Integer, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, T::AbstractMatrix{$elty})
        require_one_based_indexing(A, B, T)
        chkstride1(A,B,T)
        m     = BlasInt(size(B, 1))
        n     = BlasInt(size(B, 2))
        l = BlasInt(l_in)
        if !(min(m,n) ≥ l ≥ 0)
            throw(DimensionMismatch("too many $l"))
        end
        if size(A) ≠ (n,n)
            throw(DimensionMismatch("A has size $(size(A)), but needs size ($n,$n)"))
        end
        if size(T) ≠ (n,n)
            throw(DimensionMismatch("T has size $(size(T)), but needs size ($n,$n)"))
        end
        lda   = BlasInt(max(1,stride(A, 2)))
        ldb   = BlasInt(max(1,stride(B, 2)))
        ldt = BlasInt(max(1,stride(T,2)))

        info  = Ref{BlasInt}()
        ccall((@blasfunc($tpqrt), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, 
                Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt},
                Ptr{BlasInt}),
                m, n, l, A, lda, B, ldb, T, ldt, info)
        chklapackerror(info[])
        A, B, T
    end
end

for (tpqrt,elty) in ((:dtpqrt_,:Float64),)
    @eval function tpqrt!(l_in::Integer, nb_in::Integer, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, T::AbstractMatrix{$elty}, work::AbstractArray{$elty})
        require_one_based_indexing(A, B, T)
        chkstride1(A,B,T)
        m     = BlasInt(size(B, 1))
        n     = BlasInt(size(B, 2))
        l = BlasInt(l_in)
        nb = BlasInt(nb_in)
        if !(min(m,n) ≥ l ≥ 0)
            throw(DimensionMismatch("too many $l"))
        end
        if size(A) ≠ (n,n)
            throw(DimensionMismatch("A has size $(size(A)), but needs size ($n,$n)"))
        end
        if size(T,2) ≠ n
            throw(DimensionMismatch("T has $(size(T,2)) columns, but needs $n"))
        end
        if !(n ≥ nb ≥ 1)
            throw(DimensionMismatch("nb is invalid"))
        end
        if length(work) ≠ nb*n
            throw(DimensionMismatch("work has length $(length(work)), but needs length $(nb*n)"))
        end
        lda   = BlasInt(max(1,stride(A, 2)))
        ldb   = BlasInt(max(1,stride(B, 2)))
        ldt = BlasInt(max(1,stride(T,2)))

        info  = Ref{BlasInt}()
        ccall((@blasfunc($tpqrt), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}),
                m, n, l, nb, A, lda, B, ldb, T, ldt, work, info)
        chklapackerror(info[])
        A, B, T
    end
end

tpqrt!(l::Integer, A::AbstractMatrix, B::AbstractMatrix, τ::AbstractVector, work::AbstractVector) =
    tpqrt!(l, 1, A, B, permutedims(τ), work)

n = 2000
A = randn(n,n)
B = randn(n,n)
τ = similar(A,n)
work = similar(τ)
@time tpqrt!(n, copy(A), copy(B), τ, work);

A = randn(n,n)
B = randn(n,n)
nb = 32
T = similar(A,nb,n)
work = similar(τ,nb*n)
@time tpqrt!(n, nb, copy(A), copy(B), T, work);


T = similar(A,n,n)
@time tpqrt2!(n, A, B, T);


A = randn(n,n)
B = randn(n,n)
T = similar(A,1,n)
work = similar(T)

A2,B2,T2 = tpqrt!(n,1,copy(A), copy(B), copy(T), work)

A2,B2,T2 = tpqrt2!(n,copy(A), copy(B), copy(T))

@time qr(A);

@time qr([Matrix(UpperTriangular(A)); Matrix(UpperTriangular(B))]).T;
@time LinearAlgebra.qrfactUnblocked!([UpperTriangular(A); UpperTriangular(B)]).τ;

A2
B2

T2



F = LinearAlgebra.qrfactUnblocked!([A; B])
F.factors
F.τ

FA = LinearAlgebra.qrfactUnblocked!(copy(A))

FA.Q'*A

B

FA.factors

# DTPQRT2
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