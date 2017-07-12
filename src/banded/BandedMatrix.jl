##
# Represent a banded matrix
# [ a_11 a_12
#   a_21 a_22 a_23
#   a_31 a_32 a_33 a_34
#        a_42 a_43 a_44  ]
# ordering the data like  (columns first)
#       [ *      a_12   a_23    a_34
#         a_11   a_22   a_33    a_44
#         a_21   a_32   a_43    *
#         a_31   a_42   *       *       ]
###
type BandedMatrix{T} <: AbstractBandedMatrix{T}
    data::Matrix{T}  # l+u+1 x n (# of columns)
    m::Int #Number of rows
    l::Int # lower bandwidth ≥0
    u::Int # upper bandwidth ≥0
    function (::Type{BandedMatrix{T}}){T}(data::Matrix{T},m,l,u)
        if size(data,1) ≠ l+u+1  && !(size(data,1) == 0 && -l > u)
            error("Data matrix must have number rows equal to number of bands")
        else
            new{T}(data,m,l,u)
        end
    end
end

# BandedSubMatrix are also banded
const ColonRange = Base.Slice{Base.OneTo{Int}} # the type used in SubArray for Colon
const BandedSubMatrix{T} = Union{
        SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}},
        SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,UnitRange{Int}}},
        SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},ColonRange}},
        SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,ColonRange}}
    }

# these are the banded matrices that are ameniable to BLAS routines
const BLASBandedMatrix{T} = Union{
        BandedMatrix{T},
        SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}},
        SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,UnitRange{Int}}},
        SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},ColonRange}},
        SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,ColonRange}}
    }


isbanded{T}(::BandedSubMatrix{T}) = true

## Constructors

BandedMatrix(data::Matrix,m::Integer,a::Integer,b::Integer) = BandedMatrix{eltype(data)}(data,m,a,b)

doc"""
    BandedMatrix(T, n, m, l, u)

returns an unitialized `n`×`m` banded matrix of type `T` with bandwidths `(l,u)`.
"""

# Use zeros to avoid unallocated entries for bigfloat
BandedMatrix{T<:BlasFloat}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) =
    BandedMatrix{T}(Matrix{T}(max(0,b+a+1),m),n,a,b)
BandedMatrix{T<:Number}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) =
    BandedMatrix{T}(zeros(T,max(0,b+a+1),m),n,a,b)
BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) =
    BandedMatrix{T}(Matrix{T}(max(0,b+a+1),m),n,a,b)



BandedMatrix{T}(::Type{T},n::Integer,a::Integer,b::Integer) = BandedMatrix(T,n,n,a,b)
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) = BandedMatrix(T,n,n+b,a,b)


BandedMatrix(data::Matrix,m::Integer,a) = BandedMatrix(data,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a) = BandedMatrix(T,n,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a) = BandedMatrix(T,n,:,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,a) = BandedMatrix(T,n,-a[1],a[end])


for MAT in (:BandedMatrix, :AbstractBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval Base.convert{V}(::Type{$MAT{V}},M::BandedMatrix) =
        BandedMatrix{V}(convert(Matrix{V},M.data),M.m,M.l,M.u)
end
function Base.convert{BM<:BandedMatrix}(::Type{BM},M::Matrix)
    ret = BandedMatrix(eltype(BM)==Any ? eltype(M) :
                        promote_type(eltype(BM),eltype(M)),size(M,1),size(M,2),size(M,1)-1,size(M,2)-1)
    for k=1:size(M,1),j=1:size(M,2)
        ret[k,j] = M[k,j]
    end
    ret
end

Base.copy(B::BandedMatrix) = BandedMatrix(copy(B.data),B.m,B.l,B.u)

Base.promote_rule{T,V}(::Type{BandedMatrix{T}},::Type{BandedMatrix{V}}) = BandedMatrix{promote_type(T,V)}



for (op,bop) in ((:(Base.rand),:brand),(:(Base.zeros),:bzeros),(:(Base.ones),:bones))
    name_str = "bzeros"
    @eval begin
        $bop{T}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) =
            BandedMatrix($op(T,max(0,b+a+1),m),n,a,b)
        $bop{T}(::Type{T},n::Integer,a::Integer,b::Integer) = $bop(T,n,n,a,b)
        $bop{T}(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) = $bop(T,n,n+b,a,b)
        $bop{T}(::Type{T},::Colon,m::Integer,a::Integer,b::Integer) = $bop(T,m+a,m,a,b)
        $bop(n::Integer,m::Integer,a::Integer,b::Integer) = $bop(Float64,n,m,a,b)
        $bop(n::Integer,a::Integer,b::Integer) = $bop(n,n,a,b)

        $bop{T}(::Type{T},n::Integer,m::Integer,a) = $bop(T,n,m,-a[1],a[end])
        $bop{T}(::Type{T},n::Number,::Colon,a) = $bop(T,n,:,-a[1],a[end])
        $bop{T}(::Type{T},::Colon,m::Integer,a) = $bop(T,:,m,-a[1],a[end])
        $bop{T}(::Type{T},n::Integer,a) = $bop(T,n,-a[1],a[end])
        $bop(n::Integer,m::Integer,a) = $bop(Float64,n,m,-a[1],a[end])
        $bop(n::Integer,a) = $bop(n,-a[1],a[end])

        $bop(B::AbstractMatrix) =
            $bop(eltype(B),size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))
    end
end

doc"""
    bzeros(T,n,m,l,u)

Creates an `n×m` banded matrix  of all zeros of type `T` with bandwidths `(l,u)`
"""
bzeros

doc"""
    bones(T,n,m,l,u)

Creates an `n×m` banded matrix  with ones in the bandwidth of type `T` with bandwidths `(l,u)`
"""
bones

doc"""
    brand(T,n,m,l,u)

Creates an `n×m` banded matrix  with random numbers in the bandwidth of type `T` with bandwidths `(l,u)`
"""
brand


"""
    beye(T,n,l,u)

`n×n` banded identity matrix of type `T` with bandwidths `(l,u)`
"""
function beye{T}(::Type{T},n::Integer,a...)
    ret=bzeros(T,n,a...)
    for k=1:n
         ret[k,k]=one(T)
    end
    ret
end
beye{T}(::Type{T},n::Integer) = beye(T,n,0,0)
beye(n::Integer) = beye(n,0,0)
beye(n::Integer,a...) = beye(Float64,n,a...)

Base.similar(B::BLASBandedMatrix) =
    BandedMatrix(eltype(B),size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))


## Abstract Array Interface

size(A::BandedMatrix) = A.m, size(A.data, 2)
size(A::BandedMatrix, k::Integer) = k <= 0 ? error("dimension out of range") :
                                    k == 1 ? A.m :
                                    k == 2 ? size(A.data, 2) : 1


bandwidth(A::BandedMatrix,k::Integer) = k==1?A.l:A.u

Base.IndexStyle{T}(::Type{BandedMatrix{T}}) = IndexCartesian()

@inline colstart(A::BandedMatrix, i::Integer) = max(i-A.u, 1)
@inline  colstop(A::BandedMatrix, i::Integer) = max(min(i+A.l, size(A, 1)), 0)
@inline rowstart(A::BandedMatrix, i::Integer) = max(i-A.l, 1)
@inline  rowstop(A::BandedMatrix, i::Integer) = max(min(i+A.u, size(A, 2)), 0)


# checks if the bands match A
function checkbandmatch{T}(A::BandedMatrix{T}, V::AbstractVector, ::Colon, j::Integer)
    for k = 1:colstart(A,j)-1
        if V[k] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
    for k = colstop(A,j)+1:size(A,1)
        if V[k] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::BandedMatrix{T}, V::AbstractVector, kr::Range, j::Integer)
    a = colstart(A, j)
    b = colstop(A, j)
    i = 0
    for v in V
        k = kr[i+=1]
        if (k < a || k > b) && v ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, ::Colon)
    for j = 1:rowstart(A,k)-1
        if V[j] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
    for j = rowstop(A,j)+1:size(A,2)
        if V[j] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Range)
    a = rowstart(A, k)
    b = rowstop(A, k)
    i = 0
    for v in V
        j = jr[i+=1]
        if (j < a || j > b) && v ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::BandedMatrix{T}, V::AbstractMatrix, kr::Range, jr::Range)
    u, l = A.u, A.l
    jj = 1
    for j in jr
        kk = 1
        for k in kr
            if !(-l ≤ j - k ≤ u) && V[kk, jj] ≠ zero(T)
                # we index V manually in column-major order
                throw(BandError(A, j-k))
            end
            kk += 1
        end
        jj += 1
    end
end

checkbandmatch(A::BandedMatrix, V::AbstractMatrix, ::Colon, ::Colon) =
    checkbandmatch(A, V, 1:size(A,1), 1:size(A,2))


# TODO
# ~ implement indexing with vectors of indices
# ~ implement scalar/vector - band - integer
# ~ implement scalar/vector - band - range


# ~~ getindex ~~

# fast method used below
@inline inbands_getindex(data::AbstractMatrix, u::Integer, k::Integer, j::Integer) =
    data[u + k - j + 1, j]

@inline inbands_getindex(A::BandedMatrix, k::Integer, j::Integer) =
    inbands_getindex(A.data, A.u, k, j)


# banded get index, used for banded matrices with other data types
@inline function banded_getindex(data::AbstractMatrix, l::Integer, u::Integer, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_getindex(data, u, k, j)
    else
        zero(eltype(data))
    end
end


# scalar - integer - integer
@inline function getindex(A::BandedMatrix, k::Integer, j::Integer)
    @boundscheck  checkbounds(A, k, j)
    banded_getindex(A.data, A.l, A.u, k, j)
end

# scalar - colon - colon
@inline getindex(A::BandedMatrix, kr::Colon, jr::Colon) = copy(A)

# ~ indexing along a band

# scalar - band - colon
@inline function getindex{T}(A::BandedMatrix{T}, b::Band)
    @boundscheck checkband(A, b)
    if b.i > 0
        vec(A.data[A.u - b.i + 1, b.i+1:min(size(A,2),size(A,1)+b.i)])
    elseif b.i == 0
        vec(A.data[A.u - b.i + 1, 1:min(size(A,2),size(A,1))])
    else # b.i < 0
        vec(A.data[A.u - b.i + 1, 1:min(size(A,2),size(A,1)+b.i)])
    end
end

@inline function view{T}(A::BandedMatrix{T}, b::Band)
    @boundscheck checkband(A, b)
    if b.i > 0
        view(A.data,A.u - b.i + 1, b.i+1:min(size(A,2),size(A,1)+b.i))
    elseif b.i == 0
        view(A.data,A.u - b.i + 1, 1:min(size(A,2),size(A,1)))
    else # b.i < 0
        view(A.data,A.u - b.i + 1, 1:min(size(A,2),size(A,1)+b.i))
    end
end

# scalar - BandRange - integer -- A[1, BandRange]
@inline getindex(A::AbstractMatrix, ::Type{BandRange}, j::Integer) = A[colrange(A, j), j]

# scalar - integer - BandRange -- A[1, BandRange]
@inline getindex(A::AbstractMatrix, k::Integer, ::Type{BandRange}) = A[k, rowrange(A, k)]


# ~ indexing along a row



# give range of data matrix corresponding to colrange/rowrange
data_colrange{T}(A::BandedMatrix{T}, i::Integer) = (max(1,A.u+2-i):min(size(A,1)+A.u-i+1,size(A.data,1))) +
                                (i-1)*size(A.data,1)

data_rowrange{T}(A::BandedMatrix{T}, i::Integer) = range((i ≤ 1+A.l ? A.u+i : (i-A.l)*size(A.data,1)) ,
                                size(A.data,1)-1 ,  # step size
                                i+A.u ≤ size(A,2) ? A.l+A.u+1 : size(A,2)-i+A.l+1)

# ~~ setindex! ~~

# ~ Special setindex methods ~

# slow fall back method
@inline inbands_setindex!(A::BandedMatrix, v, k::Integer, j::Integer) =
    inbands_setindex!(A.data, A.u, v, k, j)

# fast method used below
@inline function inbands_setindex!{T}(data::AbstractMatrix{T}, u::Integer, v, k::Integer, j::Integer)
    @inbounds data[u + k - j + 1, j] = convert(T, v)::T
    v
end

@inline function banded_setindex!(data::AbstractMatrix, l::Int, u::Int, v, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_setindex!(data, u, v, k, j)
    elseif v ≠ 0  # allow setting outside bands to zero
        throw(BandError(BandedMatrix(data,size(data,2)+l,l,u),j-k))
    else # v == 0
        v
    end
end


# scalar - integer - integer
@inline function setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    @boundscheck  checkbounds(A, k, j)
    banded_setindex!(A.data, A.l, A.u, v, k ,j)
end

# scalar - colon - colon
function setindex!{T}(A::BandedMatrix{T}, v, ::Colon, ::Colon)
    if v == zero(T)
        @inbounds A.data[:] = convert(T, v)::T
    else
        throw(BandError(A, A.u+1))
    end
end

# scalar - colon
function setindex!{T}(A::BandedMatrix{T}, v, ::Colon)
    if v == zero(T)
        @inbounds A.data[:] = convert(T, v)::T
    else
        throw(BandError(A, A.u+1))
    end
end

# matrix - colon - colon
@inline function setindex!{T}(A::BandedMatrix{T}, v::AbstractMatrix, kr::Colon, jr::Colon)
    @boundscheck checkdimensions(size(A), size(v))
    @boundscheck checkbandmatch(A, v, kr, jr)

    for j=1:size(A,2), k=colrange(A,j)
        @inbounds A[k,j] = v[k,j]
    end
    A
end


function setindex!{T}(A::BandedMatrix{T}, v::AbstractVector, ::Colon)
    A[:, :] = reshape(v,size(A))
end


# ~ indexing along a band

# scalar - band - colon
@inline function setindex!{T}(A::BandedMatrix{T}, v, b::Band)
    @boundscheck checkband(A, b)
    @inbounds A.data[A.u - b.i + 1, :] = convert(T, v)::T
end

# vector - band - colon
@inline function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, b::Band)
    @boundscheck checkband(A, b)
    @boundscheck checkdimensions(diaglength(A, b), V)
    row = A.u - b.i + 1
    data, i = A.data, max(b.i + 1, 1)
    for v in V
        @inbounds data[row, i] = convert(T, v)::T
        i += 1
    end
    V
end


# ~ indexing along columns

# scalar - colon - integer -- A[:, 1] = 2 - not allowed
function setindex!{T}(A::BandedMatrix{T}, v, kr::Colon, j::Integer)
    if v == zero(T)
        A.data[:,j] = convert(T, v)::T
    else
        throw(BandError(A, _firstdiagcol(A, j)))
    end
end


# vector - colon - integer -- A[:, 1] = [1, 2, 3] - not allowed
@inline function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, kr::Colon, j::Integer)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(1:size(A,1), V)
    @boundscheck checkbandmatch(A,V,:,j)

    A.data[data_colrange(A,j)] = V[colrange(A,j)]
    V
end

# scalar - BandRange - integer -- A[1, BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, v, ::Type{BandRange}, j::Integer) =
    (A[colrange(A, j), j] = convert(T, v)::T) # call range method

# vector - BandRange - integer -- A[1, BandRange] = 2
setindex!(A::BandedMatrix, V::AbstractVector, ::Type{BandRange}, j::Integer) =
    (A[colrange(A, j), j] = V) # call range method

# scalar - range - integer -- A[1:2, 1] = 2
@inline function setindex!(A::BandedMatrix, v, kr::Range, j::Integer)
    @boundscheck checkbounds(A, kr, j)

    if v ≠ zero(eltype(A))
        @boundscheck  checkband(A, kr, j)
        data, u = A.data, A.u
        for k in kr
            inbands_setindex!(data, u, v, k, j)
        end
    else
        for k in kr ∩ colrange(A, j)
            inbands_setindex!(data, u, v, k, j)
        end
    end
    v
end

# vector - range - integer -- A[1:3, 1] = [1, 2, 3]
@inline function setindex!(A::BandedMatrix, V::AbstractVector, kr::Range, j::Integer)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(kr, V)
    @boundscheck checkbandmatch(A, V, kr, j)

    a = colstart(A, j)
    b = colstop(A, j)

    data, u, i = A.data, A.u, 0
    for v in V
        k = kr[i+=1]
        if a ≤ k ≤ b
            inbands_setindex!(data, u, v, k, j)
        end
    end
    V
end


# ~ indexing along a row

# scalar - integer - colon -- A[1, :] = 2 - not allowed
function setindex!{T}(A::BandedMatrix{T}, v, k::Integer, jr::Colon)
    if v == zero(T)
        for j in rowrange(A, k)
            inbands_setindex!(A, v, k, j)
        end
        v
    else
        throw(BandError(A, _firstdiagrow(A, k)))
    end
end

# vector - integer - colon -- A[1, :] = [1, 2, 3] - not allowed
function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Colon)
    if k < 1 || k > size(A,1)
        throw(BoundsError(A, (k, jr)))
    end
    if size(A,2) ≠ length(V)
        throw(DimensionMismatch("tried to assign $(length(V)) vector to $(size(A,1)) destination"))
    end

    for j = 1:rowstart(A,k)-1
        if V[j] ≠ zero(T)
            throw(BandError(A, _firstdiagrow(A, k)))
        end
    end
    for j = rowstop(A,j)+1:size(A,2)
        if V[j] ≠ zero(T)
            throw(BandError(A, _firstdiagrow(A, k)))
        end
    end

    A.data[data_rowrange(A,k)] = V[rowrange(A,k)]
    V
end

# scalar - integer - BandRange -- A[1, BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, v, k::Integer, ::Type{BandRange}) =
    (A[k, rowrange(A, k)] = convert(T, v)::T) # call range method

# vector - integer - BandRange -- A[1, BandRange] = [1, 2, 3]
setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, ::Type{BandRange}) =
    (A[k, rowstart(A, k):rowstop(A, k)] = V) # call range method

# scalar - integer - range -- A[1, 2:3] = 3
@inline function setindex!{T}(A::BandedMatrix{T}, v, k::Integer, jr::Range)
    @boundscheck checkbounds(A, k, jr)
    if v == zero(T)
        data, u = A.data, A.u
        for j in rowrange(A, k) ∩ jr
            inbands_setindex!(data, u, v, k, j)
        end
        v
    else
        @boundscheck checkband(A, k, jr)
        data, u = A.data, A.u
        for j in jr
            inbands_setindex!(data, u, v, k, j)
        end
    end

    v
end

# vector - integer - range -- A[1, 2:3] = [3, 4]
@inline function setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, jr::Range)
    @boundscheck checkbounds(A, k, jr)
    @boundscheck checkdimensions(jr, V)
    @boundscheck checkbandmatch(A, V, k, jr)

    a = rowstart(A, k)
    b = rowstop(A, k)

    data, u, i = A.data, A.u, 0
    for v in V
        j = jr[i+=1]
        if a ≤ j ≤ b
            inbands_setindex!(data, u, v, k, j)
        end
    end
    V
end

# ~ indexing over a rectangular block

# scalar - range - range
@inline function setindex!(A::BandedMatrix, v, kr::Range, jr::Range)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkband(A, kr, jr)
    data, u = A.data, A.u
    for j in jr, k in kr
        inbands_setindex!(data, u, v, k, j)
    end
    v
end

# matrix - range - range
@inline function setindex!(A::BandedMatrix, V::AbstractMatrix, kr::Range, jr::Range)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkdimensions(kr, jr, V)
    @boundscheck checkbandmatch(A, V, kr, jr)

    data, u, l = A.data, A.u, A.l
    jj = 1
    for j in jr
        kk = 1
        for k in kr
            if -l ≤ j - k ≤ u
                # we index V manually in column-major order
                @inbounds inbands_setindex!(data, u, V[kk, jj], k, j)
                kk += 1
            end
        end
        jj += 1
    end
    V
end

# scalar - BandRange -- A[BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, v, ::Type{BandRange}) =
    @inbounds A.data[:] = convert(T, v)::T

# ~~ end setindex! ~~



function Base.convert(::Type{Matrix},A::BandedMatrix)
    ret=zeros(eltype(A),size(A,1),size(A,2))
    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j] = A[k,j]
    end
    ret
end

Base.full(A::BandedMatrix) = convert(Matrix,A)


function Base.sparse(B::BandedMatrix)
    i=Vector{Int}(length(B.data));j=Vector{Int}(length(B.data))
    n,m=size(B.data)
    Bn=size(B,1)
    vb=copy(vec(B.data))
    for κ=1:n,ℓ=1:m
        j[κ+n*(ℓ-1)]=ℓ
        ii=κ+ℓ-B.u-1
        if ii <1 || ii > Bn
            vb[κ+n*(ℓ-1)] = 0
        end
        i[κ+n*(ℓ-1)]=min(max(ii,1),Bn)
    end

    sparse(i,j,vb,Bn,m)
end




# pass standard routines to full matrix

Base.norm(B::BandedMatrix,opts...) = norm(full(B),opts...)


# We turn off bound checking to allow nicer syntax without branching
#setindex!(A::BandedMatrix,v,k::Integer,j::Integer)=((A.l≤j-k≤A.u)&&k≤A.n)?ussetindex!(A,v,k,j):throw(BoundsError())
#setindex!(A::BandedMatrix,v,kr::Range,j::Integer)=(A.l≤j-kr[end]≤j-kr[1]≤A.u&&kr[end]≤A.n)?ussetindex!(A,v,kr,j):throw(BoundsError())


## ALgebra and other functions

function Base.scale!(α::Number, A::BandedMatrix)
    Base.scale!(α, A.data)
    A
end

function Base.scale!(A::BandedMatrix, α::Number)
    Base.scale!(A.data, α)
    A
end


function +{T,V}(A::BandedMatrix{T},B::BandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(1.,B,ret)

    ret
end

function -{T,V}(A::BandedMatrix{T}, B::BandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(-1.,B,ret)

    ret
end


function *(A::BandedMatrix, B::BandedMatrix)
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m = size(A,1),size(B,2)

    ret = BandedMatrix(promote_type(eltype(A),eltype(B)),n,m,A.l+B.l,A.u+B.u)
    for j = 1:size(ret,2), k = colrange(ret,j)
        νmin = max(1,k-bandwidth(A,1),j-bandwidth(B,2))
        νmax = min(size(A,2),k+bandwidth(A,2),j+bandwidth(B,1))

        ret[k,j] = A[k,νmin]*B[νmin,j]
        for ν=νmin+1:νmax
            ret[k,j] += A[k,ν]*B[ν,j]
        end
    end

    ret
end



function *{T<:Number,V<:Number}(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V})
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("*"))
    end
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    n,m = size(A,1),size(B,2)
    Y = BandedMatrix(promote_type(T,V),n,m,Al+Bl,Au+Bu)
    A_mul_B!(Y,A,B)
end

function *{T<:Number,V<:Number}(A::BLASBandedMatrix{T},B::StridedMatrix{V})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)

    A_mul_B!(Matrix{promote_type(T,V)}(n,m),A,B)
end

*{T<:Number,V<:Number}(A::StridedMatrix{T},B::BLASBandedMatrix{V}) =
    A*Array(B)

*{T<:BlasFloat}(A::BLASBandedMatrix{T},b::StridedVector{T}) =
    A_mul_B!(Vector{T}(size(A,1)),A,b)

function *{T}(A::BandedMatrix{T},b::StridedVector{T})
    ret = zeros(T,size(A,1))
    for j = 1:size(A,2), k = colrange(A,j)
        @inbounds ret[k]+=A[k,j]*b[j]
    end
    ret
end


function *{TT}(A::BLASBandedMatrix{TT},b::StridedVector)
    T=promote_type(eltype(A),eltype(b))
    convert(BandedMatrix{T},A)*convert(AbstractVector{T},b)
end

function Base.transpose(B::BandedMatrix)
    Bt=bzeros(eltype(B),size(B,2),size(B,1),B.u,B.l)
    for j = 1:size(B,2), k = colrange(B,j)
       Bt[j,k]=B[k,j]
    end
    Bt
end

function Base.ctranspose(B::BandedMatrix)
    Bt=bzeros(eltype(B),size(B,2),size(B,1),B.u,B.l)
    for j = 1:size(B,2), k = colrange(B,j)
       Bt[j,k]=conj(B[k,j])
    end
    Bt
end



function Base.diag{T}(A::BandedMatrix{T})
    n=size(A,1)
    @assert n==size(A,2)

    vec(A.data[A.u+1,1:n])
end



## Matrix.*Matrix

function broadcast(::typeof(*), A::BandedMatrix, B::BandedMatrix)
    @assert size(A,1)==size(B,1)&&size(A,2)==size(B,2)

    l=min(A.l,B.l);u=min(A.u,B.u)
    T=promote_type(eltype(A),eltype(B))
    ret=BandedMatrix(T,size(A,1),size(A,2),l,u)

    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j]=A[k,j]*B[k,j]
    end
    ret
end



## numbers
for OP in (:*,:/)
    @eval begin
        $OP(A::BandedMatrix, b::Number) = BandedMatrix($OP(A.data,b),A.m,A.l,A.u)
        broadcast(::typeof($OP), A::BandedMatrix, b::Number) =
            BandedMatrix($OP.(A.data,b),A.m,A.l,A.u)
    end
end


*(a::Number,B::BandedMatrix) = BandedMatrix(a*B.data,B.m,B.l,B.u)
broadcast(::typeof(*), a::Number, B::BandedMatrix) = BandedMatrix(a.*B.data,B.m,B.l,B.u)


## UniformScaling

function +(A::BandedMatrix, B::UniformScaling)
    ret = deepcopy(A)
    BLAS.axpy!(1,B,ret)
end

+(A::UniformScaling, B::BandedMatrix) = B+A

function -(A::BandedMatrix, B::UniformScaling)
    ret = deepcopy(A)
    BLAS.axpy!(-1,B,ret)
end

function -(A::UniformScaling, B::BandedMatrix)
    ret = deepcopy(B)
    Base.scale!(ret,-1)
    BLAS.axpy!(1,A,ret)
end


#implements fliplr(flipud(A))
function fliplrud(A::BandedMatrix)
    n,m=size(A)
    l=A.u+n-m
    u=A.l+m-n
    ret=BandedMatrix(eltype(A),n,m,l,u)
    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j] = A[n-k+1,m-j+1]
    end
    ret
end


for OP in (:(Base.real),:(Base.imag))
    @eval $OP(A::BandedMatrix) =
        BandedMatrix($OP(A.data),A.m,A.l,A.u)
end



## SubArray routines
# gives the band which is diagonal for the parent
bandshift(a::Range,b::Range) = first(a)-first(b)
bandshift(::ColonRange,b::Range) = 1-first(b)
bandshift(a::Range,::ColonRange) = first(a)-1
bandshift(::ColonRange,b::ColonRange) = 0
bandshift(S) = bandshift(parentindexes(S)[1],parentindexes(S)[2])

bandwidth{T}(S::BandedSubMatrix{T}, k::Integer) = bandwidth(parent(S),k) + (k==1?-1:1)*bandshift(S)


inbands_getindex{T}(S::BandedSubMatrix{T},k,j) = inbands_getindex(parent(S),parentindexes(S)[1][k],parentindexes(S)[2][j])

function Base.convert{T}(::Type{BandedMatrix},S::BandedSubMatrix{T})
    A=parent(S)
    kr,jr=parentindexes(S)
    shft=kr[1]-jr[1]
    l,u=bandwidths(A)
    if -u ≤ shft ≤ l
        BandedMatrix(A.data[:,jr],length(kr),l-shft,u+shft)
    elseif shft > l
        # need to add extra zeros at top since negative bandwidths not supported
        # new bandwidths = (0,u+shft)
        dat = zeros(T,u+shft+1,length(jr))
        dat[1:l+u+1,:] = A.data[:,jr]
        BandedMatrix(dat,length(kr),0,u+shft)
    else  # shft < -u
        dat = zeros(T,l-shft+1,length(jr))
        dat[-shft-u+1:end,:] = A.data[:,jr]  # l-shft+1 - (-shft-u) == l+u+1
        BandedMatrix(dat,length(kr),l-shft,0)
    end
end

## These routines give access to the necessary information to call BLAS

@inline leadingdimension(B::BandedMatrix) = stride(B.data,2)
@inline leadingdimension{T}(B::BandedSubMatrix{T}) = leadingdimension(parent(B))


@inline Base.pointer(B::BandedMatrix) = pointer(B.data)
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,ColonRange}}) =
    pointer(parent(B))
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},ColonRange}}) =
    pointer(parent(B))
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{ColonRange,UnitRange{Int}}}) =
    pointer(parent(B))+leadingdimension(parent(B))*(first(parentindexes(B)[2])-1)*sizeof(T)
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}}) =
    pointer(parent(B))+leadingdimension(parent(B))*(first(parentindexes(B)[2])-1)*sizeof(T)
