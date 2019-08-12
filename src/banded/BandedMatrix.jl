##
# Represent a banded matrix
# [ a_11 a_12
#   a_21 a_22 a_23
#   a_31 a_32 a_33 a_34
#        a_42 a_43 a_44  ]
# ordering the data like  (cobbndsmns first)
#       [ *      a_12   a_23    a_34
#         a_11   a_22   a_33    a_44
#         a_21   a_32   a_43    *
#         a_31   a_42   *       *       ]
###

function _BandedMatrix end

mutable struct BandedMatrix{T, CONTAINER, RAXIS} <: AbstractBandedMatrix{T}
    data::CONTAINER  # l+u+1 x n (# of columns)
    raxis::RAXIS # axis for rows (col axis comes from data)
    l::Int # lower bandwidth ≥0
    u::Int # upper bandwidth ≥0
    global function _BandedMatrix(data::AbstractMatrix{T}, raxis::AbstractUnitRange, l, u) where {T}
        if size(data,1) ≠ l+u+1  && !(size(data,1) == 0 && -l > u)
           error("Data matrix must have number rows equal to number of bands")
        else
            new{T, typeof(data), typeof(raxis)}(data, raxis, l, u)
        end
    end
end

_BandedMatrix(data::AbstractMatrix, m::Integer, l, u) = _BandedMatrix(data, Base.OneTo(m), l, u)

MemoryLayout(A::Type{BandedMatrix{T,Cont,Axes}}) where {T,Cont,Axes} = BandedColumns{typeof(MemoryLayout(Cont))}()


## Constructors

"""
    BandedMatrix{T}(undef, (n, m), (l, u))

returns an uninitialized `n`×`m` banded matrix of type `T` with bandwidths `(l,u)`.
"""
BandedMatrix{T, C}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer) where {T<:BlasFloat, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m), n, a, b)
BandedMatrix{T, C}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer) where {T, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m),n,a,b)
BandedMatrix{T, C}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer)  where {T<:Number, C<:AbstractMatrix{T}} =
    _BandedMatrix(fill!(similar(C, max(0,b+a+1),m), zero(T)),n,a,b)
BandedMatrix{T, C}(::UndefInitializer, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}) where {T, C<:AbstractMatrix{T}} =
    BandedMatrix{T,C}(undef, nm..., ab...)
BandedMatrix{T, C}(::UndefInitializer, n::Integer, ::Colon, a::Integer, b::Integer)  where {T, C<:AbstractMatrix{T}} =
    BandedMatrix{T,C}(undef,n,n+b,a,b)

BandedMatrix{T}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer)  where {T} =
    BandedMatrix{T, Matrix{T}}(undef,n,m,a,b)
BandedMatrix{T}(::UndefInitializer, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}) where T =
    BandedMatrix{T}(undef, nm..., ab...)
BandedMatrix{T}(::UndefInitializer, nm::NTuple{2,OneTo{Int}}, ab::NTuple{2,Integer}) where T =
    BandedMatrix{T}(undef, length.(nm), ab)
BandedMatrix{T}(::UndefInitializer, n::Integer, ::Colon, a::Integer, b::Integer)  where {T} =
    BandedMatrix{T}(undef,n,n+b,a,b)


BandedMatrix{V}(M::BandedMatrix) where {V} =
        _BandedMatrix(AbstractMatrix{V}(M.data), M.raxis, M.l, M.u)
BandedMatrix(M::BandedMatrix{V}) where {V} =
        _BandedMatrix(AbstractMatrix{V}(M.data), M.raxis, M.l, M.u)

convert(::Type{BandedMatrix{V}}, M::BandedMatrix{V}) where {V} = M
convert(::Type{BandedMatrix{V}}, M::BandedMatrix) where {V} =
        _BandedMatrix(convert(AbstractMatrix{V}, M.data), M.raxis, M.l, M.u)
convert(::Type{BandedMatrix}, M::BandedMatrix) = M
function convert(::Type{BandedMatrix{<:, C}}, M::BandedMatrix) where C
    M.data isa C && return M
    _BandedMatrix(convert(C, M.data), M.raxis, M.l, M.u)
end
function convert(BM::Type{BandedMatrix{T, C}}, M::BandedMatrix) where {T, C <: AbstractMatrix{T}}
    M.data isa C && return M
    _BandedMatrix(convert(C, M.data), M.raxis, M.l, M.u)
end

for MAT in (:AbstractBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval begin
        convert(::Type{$MAT{T}}, M::BandedMatrix) where {T} = convert(BandedMatrix{T}, M)
        convert(::Type{$MAT}, M::BandedMatrix) = M
        $MAT{T}(M::BandedMatrix) where {T} = BandedMatrix{T}(M)
        $MAT(M::BandedMatrix{T}) where {T} = BandedMatrix{T}(M)
    end
end

function convert(BM::Type{BandedMatrix{<:, C}}, M::AbstractMatrix) where {C}
    Container = typeof(convert(C, similar(M, 0, 0)))
    T = eltype(Container)
    ret = BandedMatrix{T, Container}(undef, size(M)..., bandwidths(M)...)
    for k=1:size(M,1),j=1:size(M,2)
        ret[k,j] = convert(T, M[k,j])
    end
    ret
end

function convert(BM::Type{BandedMatrix{T, C}}, M::AbstractMatrix) where {T, C}
    Container = typeof(convert(C, similar(M, T, 0, 0)))
    ret = BandedMatrix{T, Container}(undef, size(M)..., bandwidths(M)...)
    for k=1:size(M,1),j=1:size(M,2)
        ret[k,j] = convert(T, M[k,j])
    end
    ret
end

convert(BM::Type{BandedMatrix{T}}, M::AbstractMatrix) where {T} =
    convert(BandedMatrix{T, typeof(similar(M, T, 0, 0))}, M)

convert(BM::Type{BandedMatrix}, M::AbstractMatrix) = convert(BandedMatrix{eltype(M)}, M)

copy(B::BandedMatrix) = _BandedMatrix(copy(B.data), B.raxis, B.l, B.u)

promote_rule(::Type{BandedMatrix{T1, C1}}, ::Type{BandedMatrix{T2, C2}}) where {T1,C1, T2,C2} =
    BandedMatrix{promote_type(T1,T2), promote_type(C1, C2)}


for (op,bop) in ((:(rand),:brand),)
    @eval begin
        $bop(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) where {T} =
            _BandedMatrix($op(T,max(0,b+a+1),m),n,a,b)
        $bop(::Type{T},n::Integer,a::Integer,b::Integer) where {T} = $bop(T,n,n,a,b)
        $bop(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) where {T} = $bop(T,n,n+b,a,b)
        $bop(::Type{T},::Colon,m::Integer,a::Integer,b::Integer) where {T} = $bop(T,m+a,m,a,b)
        $bop(n::Integer,m::Integer,a::Integer,b::Integer) = $bop(Float64,n,m,a,b)
        $bop(n::Integer,a::Integer,b::Integer) = $bop(n,n,a,b)

        $bop(::Type{T},n::Integer,m::Integer,a) where {T} = $bop(T,n,m,-a[1],a[end])
        $bop(::Type{T},n::Number,::Colon,a) where {T} = $bop(T,n,:,-a[1],a[end])
        $bop(::Type{T},::Colon,m::Integer,a) where {T} = $bop(T,:,m,-a[1],a[end])
        $bop(::Type{T},n::Integer,a) where {T} = $bop(T,n,-a[1],a[end])
        $bop(n::Integer,m::Integer,a) = $bop(Float64,n,m,-a[1],a[end])
        $bop(n::Integer,a) = $bop(n,-a[1],a[end])

        $bop(B::AbstractMatrix) =
            $bop(eltype(B),size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))
    end
end


"""
    brand(T,n,m,l,u)

Creates an `n×m` banded matrix  with random numbers in the bandwidth of type `T` with bandwidths `(l,u)`
"""
brand

## Conversions from AbstractArrays, we include FillArrays in case `zeros` is ever faster
BandedMatrix{T}(A::AbstractMatrix, bnds::NTuple{2,Integer}) where T =
    BandedMatrix{T, Matrix{T}}(A, bnds)
function BandedMatrix{T, C}(A::AbstractMatrix, bnds::NTuple{2,Integer}) where {T, C <: AbstractMatrix{T}}
    (n,m) = size(A)
    (l,u) = bnds
    ret = BandedMatrix{T, C}(undef, n, m, l, u)
    @inbounds for j = 1:m, k = max(1,j-u):min(n,j+l)
        inbands_setindex!(ret, A[k,j], k, j)
    end
    ret
end

BandedMatrix(A::AbstractMatrix{T}, bnds::NTuple{2,Integer}) where T =
    BandedMatrix{T}(A, bnds)



BandedMatrix{V}(Z::Zeros{T,2}, bnds::NTuple{2,Integer}) where {T,V} =
    _BandedMatrix(zeros(V,max(0,sum(bnds)+1),size(Z,2)),size(Z,1),bnds...)

BandedMatrix(E::Eye{T}, bnds::NTuple{2,Integer}) where T = BandedMatrix{T}(E, bnds)
function BandedMatrix{T}(E::Eye, bnds::NTuple{2,Integer}) where T
    ret=BandedMatrix(Zeros{T}(E), bnds)
    ret[band(0)] .= one(T)
    ret
end

BandedMatrix{T,C}(A::AbstractMatrix) where {T, C<:AbstractMatrix{T}} = BandedMatrix{T,C}(A, bandwidths(A))
BandedMatrix{T}(A::AbstractMatrix) where T = BandedMatrix{T}(A, bandwidths(A))
BandedMatrix(A::AbstractMatrix) = BandedMatrix(A, bandwidths(A))


"""
    BandedMatrix{T}(kv::Pair, (m,n), (l,u))

Construct a m × n BandedMatrix with bandwidths (l,u)
from `Pair`s of diagonals and vectors.
Vector `kv.second` will be placed on the `kv.first` diagonal.
"""
function BandedMatrix{T}(kv::Tuple{Vararg{Pair{<:Integer,<:AbstractVector}}},
                         mn::NTuple{2,Integer},
                         lu::NTuple{2,Integer}) where T
    m,n = mn
    l,u = lu
    data = zeros(T, u+l+1, n)
    for (k,v) in kv
        p = length(v)
        if k ≤ 0
            data[u-k+1,1:p] = v
        else
            data[u-k+1,k+1:k+p] = v
        end
    end

    return _BandedMatrix(data, m, l, u)
end

BandedMatrix(kv::Tuple{Vararg{Pair{<:Integer,<:AbstractVector}}},
                         mn::NTuple{2,Integer},
                         lu::NTuple{2,Integer}) =
    BandedMatrix{promote_type(map(x -> eltype(x.second), kv)...)}(kv, mn, lu)



function BandedMatrix{T}(kv::Tuple{Vararg{Pair{<:Integer,<:AbstractVector}}},
                         nm::NTuple{2,Integer}) where T
    u = mapreduce(x -> x.first, max, kv)
    l = -mapreduce(x -> x.first, min, kv)
    BandedMatrix{T}(kv, nm, (l,u))
end

BandedMatrix(kv::Tuple{Vararg{Pair{<:Integer,<:AbstractVector}}}, nm::NTuple{2,Integer}) =
    BandedMatrix{promote_type(map(x -> eltype(x.second), kv)...)}(kv, nm)



function BandedMatrix{T}(kv::Pair{<:Integer,<:AbstractVector}...) where T
    n = mapreduce(x -> length(x.second) + abs(x.first), max, kv)
    u = mapreduce(x -> x.first, max, kv)
    l = -mapreduce(x -> x.first, min, kv)
    BandedMatrix{T}(kv, (n,n), (l,u))
end

"""
    BandedMatrix(kv::Pair{<:Integer,<:AbstractVector}...)

Construct a square matrix from `Pair`s of diagonals and vectors.
Vector `kv.second` will be placed on the `kv.first` diagonal.
"""
BandedMatrix(kv::Pair{<:Integer,<:AbstractVector}...) =
    BandedMatrix{promote_type(map(x -> eltype(x.second), kv)...)}(kv...)



"""
    similar(bm::BandedMatrix, [T::Type], [n::Integer, m::Integer, [l::Integer, u::Integer]])
    similar(bm::BandedSubBandedMatrix, [T::Type], [n::Integer, m::Integer, [l::Integer, u::Integer]])

Creates a banded matrix similar in type to the input. Where the eltype `T`, the
sizes `n`, and `m`, the band limits `l`, and `u` are not provided, they default
to the values in the input banded matrix.
"""
function similar(bm::BandedMatrix, T::Type=eltype(bm),
                      n::Integer=size(bm, 1), m::Integer=size(bm, 2),
                      l::Integer=bandwidth(bm, 1), u::Integer=bandwidth(bm, 2))
    data = similar(bm.data, T, max(0, u+l+1), m)
    _BandedMatrix(data, n, l, u)
end

similar(bm::AbstractBandedMatrix, n::Integer, m::Integer) = similar(bm, eltype(bm), m, n)
similar(bm::AbstractBandedMatrix, n::Integer, m::Integer, l::Integer, u::Integer) =
    similar(bm, eltype(bm), m, n, l, u)



## Abstract Array Interface

axes(A::BandedMatrix) = (A.raxis, axes(A.data,2))
size(A::BandedMatrix) = (length(A.raxis), size(A.data,2))


## banded matrix interface
bandeddata(A::BandedMatrix) = A.data
bandwidths(A::BandedMatrix) = (A.l , A.u)

IndexStyle(::Type{BandedMatrix{T}}) where {T} = IndexCartesian()

# ~~ getindex ~~

# fast method used below
@inline inbands_getindex(data::AbstractMatrix, u::Integer, k::Integer, j::Integer) =
    data[u + k - j + 1, j]

@inline function inbands_getindex(A::BandedMatrix, k::Integer, j::Integer)
    # it takes a bit of time to extract A.data, A.u since julia checks if those fields exist
    # the @inbounds here will suppress those checks
    @inbounds r = inbands_getindex(A.data, A.u, k, j)
    r
end

# work around for Any matrices
_zero(T) = zero(T)
_zero(::Type{Any}) = nothing

# banded get index, used for banded matrices with other data types
@inline function banded_getindex(data::AbstractMatrix, l::Integer, u::Integer, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_getindex(data, u, k, j)
    else
        _zero(eltype(data)) 
    end
end


# scalar - integer - integer
@inline function getindex(A::BandedMatrix, k::Integer, j::Integer)
    @boundscheck checkbounds(A, k, j)
    @inbounds r = banded_getindex(A.data, A.l, A.u, k, j)
    r
end

# ~ indexing along a band
# we reduce it to converting a View


# type to represent a view of a band
const BandedMatrixBand{T} = SubArray{T, 1, ReshapedArray{T,1,BandedMatrix{T},
                                Tuple{MultiplicativeInverses.SignedMultiplicativeInverse{Int}}}, Tuple{BandSlice}, false}


band(V::BandedMatrixBand) = first(parentindices(V)).band.i

# gives a view of the parent's data matrix
function dataview(V::BandedMatrixBand)
    A = parent(parent(V))
    b = band(V)
    m,n = size(A)
    if b > 0
        view(A.data, A.u - b + 1, b+1:min(n,m+b))
    elseif b == 0
        view(A.data, A.u - b + 1, 1:min(n,m))
    else # b < 0
        view(A.data, A.u - b + 1, 1:min(n,m+b))
    end
end

function convert(::Type{Vector{T}}, V::BandedMatrixBand) where T
    A = parent(parent(V))
    if -A.l ≤ band(V) ≤ A.u
        Vector{T}(dataview(V))
    else
        zeros(T, length(V))
    end
end

convert(::Type{Array{T}}, A::BandedMatrixBand) where T = convert(Vector{T}, A)
convert(::Type{Array}, A::BandedMatrixBand) = convert(Vector{eltype(A)}, A)
convert(::Type{Vector}, A::BandedMatrixBand)= convert(Vector{eltype(A)}, A)


convert(::Type{AbstractArray{T}}, A::BandedMatrixBand{T}) where T = A
convert(::Type{AbstractVector{T}}, A::BandedMatrixBand{T}) where T = A
convert(::Type{AbstractArray}, A::BandedMatrixBand{T}) where T = A
convert(::Type{AbstractVector}, A::BandedMatrixBand{T}) where T = A

convert(::Type{AbstractArray{T}}, A::BandedMatrixBand) where T = convert(Vector{T}, A)
convert(::Type{AbstractVector{T}}, A::BandedMatrixBand) where T = convert(Vector{T}, A)



# ~ indexing along a row



# give range of data matrix corresponding to colrange/rowrange
data_colrange(A::BandedMatrix{T}, i::Integer) where {T} =
    (max(1,A.u+2-i):min(size(A,1)+A.u-i+1,size(A.data,1))) .+
                                ((i-1)*size(A.data,1))

data_rowrange(A::BandedMatrix{T}, i::Integer) where {T} = range((i ≤ 1+A.l ? A.u+i : (i-A.l)*size(A.data,1)) ,
                                size(A.data,1)-1 ,  # step size
                                i+A.u ≤ size(A,2) ? A.l+A.u+1 : size(A,2)-i+A.l+1)

# ~~ setindex! ~~

# ~ Special setindex methods ~

# fast method used below
@inline @propagate_inbounds function inbands_setindex!(data::AbstractMatrix{T}, u::Integer, v, k::Integer, j::Integer) where {T}
    data[u + k - j + 1, j] = convert(T, v)::T
    v
end

# slow fall back method
@inline @propagate_inbounds function inbands_setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    # it takes a bit of time to extract A.data, A.u since julia checks if those fields exist
    # the @inbounds here will suppress those checks
    @inbounds r = inbands_setindex!(A.data, A.u, v, k, j)
    r
end

@inline @propagate_inbounds function banded_setindex!(data::AbstractMatrix, l::Integer, u::Integer, v, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_setindex!(data, u, v, k, j)
    elseif v ≠ 0  # allow setting outside bands to zero
        throw(BandError(data,j-k))
    else # v == 0
        v
    end
end

# scalar - integer - integer
@inline @propagate_inbounds function setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    @boundscheck checkbounds(A, k, j)
    @inbounds r = banded_setindex!(A.data, A.l, A.u, v, k ,j)
    r
end

# matrix - colon - colon
@inline @propagate_inbounds function setindex!(A::BandedMatrix{T}, v::AbstractMatrix, kr::Colon, jr::Colon) where {T}
    @boundscheck checkdimensions(size(A), size(v))
    @boundscheck checkbandmatch(A, v, kr, jr)

    for j=1:size(A,2), k=colrange(A,j)
        @inbounds A[k,j] = v[k,j]
    end
    A
end

@propagate_inbounds function setindex!(A::BandedMatrix{T}, v::AbstractVector, ::Colon) where {T}
    A[:, :] = reshape(v,size(A))
end


# ~ indexing along a band

# vector - band - colon
@inline @propagate_inbounds function setindex!(A::BandedMatrix{T}, V::AbstractVector, b::Band) where {T}
    @boundscheck checkband(A, b)
    @boundscheck checkdimensions(diaglength(A, b), V)
    row = A.u - b.i + 1
    data, i = A.data, max(b.i + 1, 1)
    for v in V
        data[row, i] = convert(T, v)::T
        i += 1
    end
    V
end


# ~ indexing along columns
# vector - colon - integer -- A[:, 1] = [1, 2, 3] - not allowed
@inline @propagate_inbounds function setindex!(A::BandedMatrix{T}, V::AbstractVector, kr::Colon, j::Integer) where {T}
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(1:size(A,1), V)
    @boundscheck checkbandmatch(A,V,:,j)

    A.data[data_colrange(A,j)] = V[colrange(A,j)]
    V
end

# vector - BandRange - integer -- A[1, BandRange] = 2
@propagate_inbounds setindex!(A::BandedMatrix, V::AbstractVector, ::BandRangeType, j::Integer) =
    (A[colrange(A, j), j] = V) # call range method

# vector - range - integer -- A[1:3, 1] = [1, 2, 3]
@inline @propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, kr::AbstractRange, j::Integer)
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

# vector - integer - colon -- A[1, :] = [1, 2, 3] - not allowed
@inline @propagate_inbounds function setindex!(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Colon) where {T}
    @boundscheck if k < 1 || k > size(A,1)
        throw(BoundsError(A, (k, jr)))
    end
    @boundscheck if size(A,2) ≠ length(V)
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

# vector - integer - BandRange -- A[1, BandRange] = [1, 2, 3]
@propagate_inbounds setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, ::BandRangeType) =
    (A[k, rowstart(A, k):rowstop(A, k)] = V) # call range method

# vector - integer - range -- A[1, 2:3] = [3, 4]
@inline @propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, jr::AbstractRange)
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

# matrix - range - range
@inline @propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractMatrix, kr::AbstractRange, jr::AbstractRange)
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
                inbands_setindex!(data, u, V[kk, jj], k, j)
                kk += 1
            end
        end
        jj += 1
    end
    V
end

# ~~ end setindex! ~~



function convert(::Type{Matrix}, A::BandedMatrix)
    ret=zeros(eltype(A),size(A,1),size(A,2))
    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j] = A[k,j]
    end
    ret
end

function _banded_colval(B::AbstractMatrix)
    data = bandeddata(B)
    j = Vector{Int}(undef, length(data))
    n,m = size(data)
    for κ=1:n, ℓ=1:m
        j[κ+n*(ℓ-1)] = ℓ
    end
    j
end

function _banded_rowval(B::AbstractMatrix)
    data = bandeddata(B)
    i = Vector{Int}(undef, length(data))
    n,m = size(data)
    Bn = size(B,1)
    l,u = bandwidths(B)
    for κ=1:n, ℓ=1:m
        ii=κ+ℓ-u-1
        i[κ+n*(ℓ-1)] = min(max(ii,1),Bn)
    end
    i
end

function _banded_nzval(B::AbstractMatrix)
    data = bandeddata(B)
    i = Vector{Int}(undef, length(data))
    n,m = size(data)
    Bn = size(B,1)
    vb = copy(vec(data))
    l,u = bandwidths(B)
    for κ=1:n, ℓ=1:m
        ii=κ+ℓ-u-1
        if ii <1 || ii > Bn
            vb[κ+n*(ℓ-1)] = 0
        end
    end
    vb
end


sparse(B::BandedMatrix) = sparse(_banded_rowval(B), _banded_colval(B), _banded_nzval(B), size(B)...)




function _bidiagonalize!(A::AbstractMatrix{T}, M::BandedColumnMajor) where T
    m, n = size(A)
    mn = min(m, n)
    d = Vector{T}(undef, mn)
    e = Vector{T}(undef, mn-1)
    Q = Matrix{T}(undef, 0, 0)
    Pt = Matrix{T}(undef, 0, 0)
    C = Matrix{T}(undef, 0, 0)
    work = Vector{T}(undef, 2*max(m, n))
    gbbrd!('N', m, n, 0, bandwidth(A, 1), bandwidth(A, 2), bandeddata(A), d, e, Q, Pt, C, work)
    Bidiagonal(d, e, :U)
end

bidiagonalize!(A::AbstractMatrix) = _bidiagonalize!(A, MemoryLayout(typeof(A)))
bidiagonalize(A::AbstractMatrix) = bidiagonalize!(copy(A))

svdvals!(A::BandedMatrix) = svdvals!(bidiagonalize!(A))
svdvals(A::BandedMatrix) = svdvals!(copy(A))

# We turn off bound checking to allow nicer syntax without branching
#setindex!(A::BandedMatrix,v,k::Integer,j::Integer)=((A.l≤j-k≤A.u)&&k≤A.n)?ussetindex!(A,v,k,j):throw(BoundsError())
#setindex!(A::BandedMatrix,v,kr::AbstractRange,j::Integer)=(A.l≤j-kr[end]≤j-kr[1]≤A.u&&kr[end]≤A.n)?ussetindex!(A,v,kr,j):throw(BoundsError())


## ALgebra and other functions

function fill!(A::BandedMatrix{T}, x) where T
    x == zero(T) || throw(BandError(A))
    fill!(A.data, x)
    A
end

function diag(A::BandedMatrix{T}) where {T}
    n=size(A,1)
    @assert n==size(A,2)

    vec(A.data[A.u+1,1:n])
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


for OP in (:real, :imag)
    @eval $OP(A::BandedMatrix) = _BandedMatrix($OP(A.data),A.raxis,A.l,A.u)
end


## BandedSubBandedMatrix routines


# getindex(A::AbstractBandedMatrix, I...) = _materialize(view(A, I...))

# gives the band which is diagonal for the parent
bandshift(a::AbstractRange, b::AbstractRange) = first(a)-first(b)
bandshift(::Slice{OneTo{Int}}, b::AbstractRange) = 1-first(b)
bandshift(a::AbstractRange, ::Slice{OneTo{Int}}) = first(a)-1
bandshift(::Slice{OneTo{Int}}, b::Slice{OneTo{Int}}) = 0
bandshift(S) = bandshift(parentindices(S)[1],parentindices(S)[2])




# BandedMatrix with unit range indexes is also banded
const BandedSubBandedMatrix{T, C, R, I1<:AbstractUnitRange, I2<:AbstractUnitRange, t} =
    SubArray{T,2,BandedMatrix{T, C, R},Tuple{I1,I2},t}

isbanded(::BandedSubBandedMatrix) = true
MemoryLayout(::Type{BandedSubBandedMatrix{T,C,R,I1,I2,t}}) where {T,C,R,I1,I2,t} = 
    BandedColumns{typeof(MemoryLayout(SubArray{T,2,C,Tuple{Slice{OneTo{Int}},I2},t}))}()
BroadcastStyle(::Type{<:BandedSubBandedMatrix}) = BandedStyle()

function _shift(bm::BandedSubBandedMatrix)
    kr,jr=parentindices(bm)
    kr[1]-jr[1]
end

function similar(bm::BandedSubBandedMatrix, T::Type=eltype(bm),
                      n::Integer=size(bm, 1), m::Integer=size(bm, 2),
                      l::Integer=max(0, bandwidth(parent(bm), 1) - _shift(bm)),
                      u::Integer=max(0, bandwidth(parent(bm), 2) + _shift(bm)))
    similar(bm.parent, T, n, m, l, u)
end

bandeddata(V::BandedSubBandedMatrix) = view(bandeddata(parent(V)), :, parentindices(V)[2])

bandwidths(S::SubArray{T,2,<:AbstractMatrix,I}) where {T,I<:Tuple{Vararg{AbstractUnitRange}}} =
    bandwidths(parent(S)) .+ (-1,1) .* bandshift(S)

if VERSION < v"1.2-"
    @inline function inbands_getindex(S::BandedSubBandedMatrix{T}, k::Integer, j::Integer) where {T}
        @inbounds r = inbands_getindex(parent(S), reindex(S, parentindices(S), (k, j))...)
        r
    end

    @inline function inbands_setindex!(S::BandedSubBandedMatrix{T}, v, k::Integer, j::Integer) where {T}
        @inbounds r = inbands_setindex!(parent(S), v, reindex(S, parentindices(S), (k, j))...)
        r
    end
else
    @inline function inbands_getindex(S::BandedSubBandedMatrix{T}, k::Integer, j::Integer) where {T}
        @inbounds r = inbands_getindex(parent(S), reindex(parentindices(S), (k, j))...)
        r
    end
    
    @inline function inbands_setindex!(S::BandedSubBandedMatrix{T}, v, k::Integer, j::Integer) where {T}
        @inbounds r = inbands_setindex!(parent(S), v, reindex(parentindices(S), (k, j))...)
        r
    end
end


function convert(::Type{BandedMatrix}, S::BandedSubBandedMatrix)
    A=parent(S)
    kr,jr=parentindices(S)
    shft=kr[1]-jr[1]
    l,u=bandwidths(A)
    if -u ≤ shft ≤ l
        data = A.data[:, jr]
    elseif shft > l
        # need to add extra zeros at top since negative bandwidths not supported
        # new bandwidths = (0,u+shft)
        data = fill!(similar(A.data,u+shft+1,length(jr)), zero(eltype(A.data)))
        data[1:l+u+1,:] = A.data[:,jr]
    else  # shft < -u
        data = fill!(similar(A.data,l-shft+1,length(jr)), zero(eltype(A.data)))
        data[-shft-u+1:end,:] = A.data[:,jr]  # l-shft+1 - (-shft-u) == l+u+1
    end
    _BandedMatrix(data,length(kr),max(0, l-shft),max(0, u+shft))
end
