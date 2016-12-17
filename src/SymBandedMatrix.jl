export sbrand, sbeye, sbzeros

#
# Represent a symmetric banded matrix
# [ a_11 a_12 a_13
#   a_12 a_22 a_23 a_24
#   a_13 a_23 a_33 a_34
#        a_24 a_34 a_44  ]
# ordering the data like  (columns first)
#       [ *      *      a_13   a_24
#         *      a_12   a_23   a_34
#         a_11   a_22   a_33   a_44 ]
###
type SymBandedMatrix{T} <: AbstractBandedMatrix{T}
    data::Matrix{T}  # k+1 x n (# of columns)
    k::Int # bandwidth ≥ 0
    function SymBandedMatrix(data::Matrix{T},k)
        if size(data,1) != k+1
            error("Data matrix must have number rows equal to number of superdiagonals")
        else
            new(data,k)
        end
    end
end


SymBandedMatrix(data::Matrix,k::Integer) = SymBandedMatrix{eltype(data)}(data,k)

doc"""
    SymBandedMatrix(T, n, k)

returns an unitialized `n`×`n` symmetric banded matrix of type `T` with bandwidths `(-k,k)`.
"""

# Use zeros to avoid unallocated entries for bigfloat
SymBandedMatrix{T<:BlasFloat}(::Type{T},n::Integer,k::Integer) =
    SymBandedMatrix{T}(Array(T,k+1,n),k)
SymBandedMatrix{T<:Number}(::Type{T},n::Integer,k::Integer) =
    SymBandedMatrix{T}(zeros(T,k+1,n),k)
SymBandedMatrix{T}(::Type{T},n::Integer,k::Integer) =
    SymBandedMatrix{T}(Array(T,k+1,n),k)


for MAT in (:SymBandedMatrix,  :AbstractBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval Base.convert{V}(::Type{$MAT{V}},M::SymBandedMatrix) =
        SymBandedMatrix{V}(convert(Matrix{V},M.data),M.k)
end

Base.copy(B::SymBandedMatrix) = SymBandedMatrix(copy(B.data),B.k)

Base.promote_rule{T,V}(::Type{SymBandedMatrix{T}},::Type{SymBandedMatrix{V}}) =
    SymBandedMatrix{promote_type(T,V)}



for (op,bop) in ((:(Base.rand),:sbrand),(:(Base.zeros),:sbzeros),(:(Base.ones),:sbones))
    @eval begin
        $bop{T}(::Type{T},n::Integer,a::Integer) = SymBandedMatrix($op(T,a+1,n),a)
        $bop(n::Integer,a::Integer) = $bop(Float64,n,a)

        $bop(B::AbstractMatrix) = $bop(eltype(B),size(B,1),bandwidth(B,2))
    end
end

doc"""
    sbzeros(T,n,k)

Creates an `n×n` symmetric banded matrix  of all zeros of type `T` with bandwidths `(k,k)`
"""
sbzeros

doc"""
    sbones(T,n,k)

Creates an `n×n` symmetric banded matrix  with ones in the bandwidth of type `T` with bandwidths `(k,k)`
"""
sbones

doc"""
    sbrand(T,n,k)

Creates an `n×n` symmetric banded matrix  with random numbers in the bandwidth of type `T` with bandwidths `(k,k)`
"""
sbrand


"""
    sbeye(T,n,l,u)

`n×n` banded identity matrix of type `T` with bandwidths `(l,u)`
"""
function sbeye{T}(::Type{T},n::Integer,a=0)
    ret=sbzeros(T,n,a)
    ret[band(0)] = one(T)
    ret
end
sbeye(n::Integer,a...) = sbeye(Float64,n,a...)



## Abstract Array Interface

Base.size(A::SymBandedMatrix, k) = size(A.data,2)
function Base.size(A::SymBandedMatrix)
    n = size(A.data,2)
    n,n
end

bandwidth(A::SymBandedMatrix,k) = A.k

Base.linearindexing{T}(::Type{SymBandedMatrix{T}}) = Base.LinearSlow()


@inline inbands_getindex(A::SymBandedMatrix, k::Integer, j::Integer) =
    A.data[A.k - abs(k-j) + 1, max(k,j)]


# banded get index, used for banded matrices with other data types
@inline function symbanded_getindex(data::AbstractMatrix, l::Integer, u::Integer, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_getindex(data, u, k, j)
    else
        zero(eltype(data))
    end
end

# scalar - integer - integer
@inline function getindex(A::SymBandedMatrix, k::Integer, j::Integer)
    @boundscheck  checkbounds(A, k, j)
    if -A.k ≤ j-k ≤ A.k
        inbands_getindex(A, k, j)
    else
        zero(eltype(A))
    end
end


# scalar - colon - colon
@inline getindex(A::SymBandedMatrix, kr::Colon, jr::Colon) = copy(A)

# ~ indexing along a band

# scalar - band - colon
@inline function getindex{T}(A::SymBandedMatrix{T}, b::Band)
    @boundscheck checkband(A, b)
    vec(A.data[A.k - abs(b.i) + 1, b.i+1:end])
end

@inline function view{T}(A::SymBandedMatrix{T}, b::Band)
    @boundscheck checkband(A, b)
    view(A.data,A.k - abs(b.i) + 1, b.i+1:size(A.data,2))
end



# ~~ setindex! ~~

# ~ Special setindex methods ~

# slow fall back method
@inline syminbands_setindex!(A::SymBandedMatrix, v, k::Integer, j::Integer) =
    syminbands_setindex!(A.data, A.k, v, k, j)

# fast method used below
@inline function syminbands_setindex!{T}(data::AbstractMatrix{T}, u::Integer, v, k::Integer, j::Integer)
    @inbounds data[u + abs(k-j) + 1, max(k,j)] = convert(T, v)::T
    v
end

@inline function symbanded_setindex!(data::AbstractMatrix, u::Int, v, k::Integer, j::Integer)
    if -u ≤ j-k ≤ u
        syminbands_setindex!(data, u, v, k, j)
    elseif v ≠ 0  # allow setting outside bands to zero
        throw(BandError(SymBandedMatrix(data,u),j-k))
    else # v == 0
        v
    end
end

# scalar - colon - colon
function setindex!{T}(A::SymBandedMatrix{T}, v, ::Colon, ::Colon)
    if v == zero(T)
        @inbounds A.data[:] = convert(T, v)::T
    else
        throw(BandError(A, A.k+1))
    end
end

function Base.convert(::Type{Matrix},A::SymBandedMatrix)
    ret=zeros(eltype(A),size(A,1),size(A,2))
    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j] = A[k,j]
    end
    ret
end

Base.full(A::SymBandedMatrix) = convert(Matrix,A)



# algebra
function +{T,V}(A::SymBandedMatrix{T},B::SymBandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A)

    ret = sbzeros(promote_type(T,V),n,max(A.k,B.k))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(1.,B,ret)

    ret
end

function -{T,V}(A::SymBandedMatrix{T}, B::SymBandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A)

    ret = sbzeros(promote_type(T,V),n,max(A.k,B.k))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(-1.,B,ret)

    ret
end


function *{T<:Number,V<:Number}(A::SymBandedMatrix{T},B::SymBandedMatrix{V})
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("*"))
    end
    Ak = bandwidth(A,2)
    Bk = bandwidth(B,2)
    n = size(A,1)
    Y = BandedMatrix(promote_type(T,V),n,Ak+Bk)
    A_mul_B!(Y,A,B)
end

function *{T<:Number,V<:Number}(A::SymBandedMatrix{T},B::StridedMatrix{V})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)

    A_mul_B!(Array(promote_type(T,V),n,m),A,B)
end

*{T<:Number,V<:Number}(A::StridedMatrix{T},B::SymBandedMatrix{V}) =
    A*Array(B)

*{T<:BlasFloat}(A::SymBandedMatrix{T},b::StridedVector{T}) =
    A_mul_B!(Array(T,size(A,1)),A,b)

function *{T}(A::SymBandedMatrix{T},b::StridedVector{T})
    ret = zeros(T,size(A,1))
    for j = 1:size(A,2), k = colrange(A,j)
        @inbounds ret[k]+=A[k,j]*b[j]
    end
    ret
end


function *{TT}(A::SymBandedMatrix{TT},b::StridedVector)
    T=promote_type(eltype(A),eltype(b))
    convert(BandedMatrix{T},A)*convert(AbstractVector{T},b)
end

Base.transpose(B::SymBandedMatrix) = copy(B)

Base.ctranspose{T<:Real}(B::SymBandedMatrix{T}) = copy(B)



Base.diag{T}(A::SymBandedMatrix{T}) = vec(A.data[A.k+1,:])



## A_***_B routines

@inline leadingdimension(B::SymBandedMatrix) = stride(B.data,2)
@inline Base.pointer(B::SymBandedMatrix) = pointer(B.data)

sbmv!{T<:BlasFloat}(α::T,A::SymBandedMatrix{T},x::StridedVector{T},β::T,y::StridedVector{T}) =
  sbmv!('U',A.k,α,A.data,x,β,y)



function symbanded_A_mul_B!{T<:BlasFloat}(c::AbstractVector{T},A::AbstractMatrix{T},b::StridedVector{T})
    n = size(A,1)

    @boundscheck if length(c) ≠ n || length(b) ≠ n
        throw(DimensionMismatch())
    end

    k = bandwidth(A,2)
    sbmv!('U',n,k,one(T),
            pointer(A),leadingdimension(A),pointer(b),stride(b,1),zero(T),pointer(c),stride(c,1))
    c
end


Base.A_mul_B!{T}(c::AbstractVector,A::SymBandedMatrix{T},b::AbstractVector) =
    symbanded_A_mul_B!(c,A,b)
