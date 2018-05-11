undef##
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

mutable struct BandedMatrix{T, CONTAINER} <: AbstractBandedMatrix{T}
    data::CONTAINER  # l+u+1 x n (# of columns)
    m::Int #Number of rows
    l::Int # lower bandwidth ≥0
    u::Int # upper bandwidth ≥0
    global function _BandedMatrix(data::AbstractMatrix{T},m,l,u) where {T}
        if size(data,1) ≠ l+u+1  && !(size(data,1) == 0 && -l > u)
           error("Data matrix must have number rows equal to number of bands")
        else
            new{T, typeof(data)}(data,m,l,u)
        end
    end
end

MemoryLayout(::BandedMatrix{T}) where T = BandedLayout{T}()


# BandedMatrix with unit range indexes is also banded
const BandedSubBandedMatrix{T, C} =
    SubArray{T,2,BandedMatrix{T, C},I} where I<:Tuple{Vararg{AbstractUnitRange}}

@banded BandedSubBandedMatrix

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
BandedMatrix{T}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer)  where {T} =
    BandedMatrix{T, Matrix{T}}(undef,n,m,a,b)
BandedMatrix{T}(::UndefInitializer, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}) where T =
    BandedMatrix{T}(undef, nm..., ab...)
BandedMatrix{T}(::UndefInitializer, n::Integer, ::Colon, a::Integer, b::Integer)  where {T} =
    BandedMatrix{T}(undef,n,n+b,a,b)


BandedMatrix{V}(M::BandedMatrix) where {V} =
        _BandedMatrix(AbstractMatrix{V}(M.data), M.m, M.l, M.u)
BandedMatrix(M::BandedMatrix{V}) where {V} =
        _BandedMatrix(AbstractMatrix{V}(M.data), M.m, M.l, M.u)

convert(::Type{BandedMatrix{V}}, M::BandedMatrix{V}) where {V} = M
convert(::Type{BandedMatrix{V}}, M::BandedMatrix) where {V} =
        _BandedMatrix(convert(AbstractMatrix{V}, M.data), M.m, M.l, M.u)
convert(::Type{BandedMatrix}, M::BandedMatrix{V}) where {V} = M

for MAT in (:AbstractBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval begin
        Base.convert(::Type{$MAT{T}}, M::BandedMatrix) where {T} = convert(BandedMatrix{T}, M)
        Base.convert(::Type{$MAT}, M::BandedMatrix{T}) where {T} = convert(BandedMatrix{T}, M)
        $MAT{T}(M::BandedMatrix) where {T} = BandedMatrix{T}(M)
        $MAT(M::BandedMatrix{T}) where T = BandedMatrix{T}(M)
    end
end


function Base.convert(BM::Type{<: BandedMatrix{T}}, M::Matrix) where {T}
    ret = BandedMatrix{T}(undef, size(M,1),size(M,2),size(M,1)-1,size(M,2)-1)
    for k=1:size(M,1),j=1:size(M,2)
        ret[k,j] = convert(T, M[k,j])
    end
    ret
end

Base.convert(BM::Type{BandedMatrix}, M::Matrix) = convert(BM{eltype(M)}, M)

Base.copy(B::BandedMatrix) = _BandedMatrix(copy(B.data), B.m, B.l, B.u)

Base.promote_rule(::Type{BandedMatrix{T1, C1}}, ::Type{BandedMatrix{T2, C2}}) where {T1,C1, T2,C2} =
    BandedMatrix{promote_type(T1,T2), promote_type(C1, C2)}


for (op,bop) in ((:(Base.rand),:brand),)
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
BandedMatrix{T}(A::AbstractMatrix, bnds::NTuple{2,Int}) where T =
    BandedMatrix{T, Matrix{T}}(A, bnds)
function BandedMatrix{T, C}(A::AbstractMatrix, bnds::NTuple{2,Int}) where {T, C <: AbstractMatrix{T}}
    (n,m) = size(A)
    (l,u) = bnds
    ret = BandedMatrix{T, C}(undef, n, m, l, u)
    @inbounds for j = 1:m, k = max(1,j-u):min(n,j+l)
        inbands_setindex!(ret, A[k,j], k, j)
    end
    ret
end

BandedMatrix(A::AbstractMatrix{T}, bnds::NTuple{2,Int}) where T =
    BandedMatrix{T}(A, bnds)

BandedMatrix{V}(Z::Zeros{T,2}, bnds::NTuple{2,Int}) where {T,V} =
    _BandedMatrix(zeros(V,max(0,sum(bnds)+1),size(Z,2)),size(Z,1),bnds...)

BandedMatrix(E::Eye{T}, bnds::NTuple{2,Int}) where T = BandedMatrix{T}(E, bnds)
function BandedMatrix{T}(E::Eye, bnds::NTuple{2,Int}) where T
    ret=BandedMatrix(Zeros{T}(E), bnds)
    ret[band(0)] = one(T)
    ret
end

BandedMatrix(A::AbstractMatrix) = BandedMatrix(A, bandwidths(A))


function BandedMatrix{T}(kv::Tuple{Vararg{Pair{<:Integer,<:AbstractVector}}},
                         mn::NTuple{2,Int},
                         lu::NTuple{2,Int}) where T
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


Base.similar(B::BandedMatrix) =
    BandedMatrix{eltype(B)}(size(B,1), size(B,2), bandwidth(B,1), bandwidth(B,2))


## Abstract Array Interface

size(A::BandedMatrix) = A.m, size(A.data, 2)
size(A::BandedMatrix, k::Integer) = k <= 0 ? error("dimension out of range") :
                                    k == 1 ? A.m :
                                    k == 2 ? size(A.data, 2) : 1


bandwidth(A::BandedMatrix,k::Integer) = k==1 ? A.l : A.u

Base.IndexStyle(::Type{BandedMatrix{T}}) where {T} = IndexCartesian()


# TODO
# ~ implement indexing with vectors of indices
# ~ implement scalar/vector - band - integer
# ~ implement scalar/vector - band - range


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
    @boundscheck checkbounds(A, k, j)
    @inbounds r = banded_getindex(A.data, A.l, A.u, k, j)
    r
end

# scalar - colon - colon
@inline getindex(A::BandedMatrix, kr::Colon, jr::Colon) = copy(A)

# ~ indexing along a band
# we reduce it to converting a View

# scalar - band - colon
@inline getindex(A::BandedMatrix{T}, b::Band) where {T} = Vector{T}(view(A, b))

# type to represent a view of a band
const BandedMatrixBand{T} = SubArray{T, 1, Base.ReshapedArray{T,1,BandedMatrix{T},
                                Tuple{Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int}}}, Tuple{BandSlice}, false}


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


# scalar - BandRange - integer -- A[1, BandRange]
@inline getindex(A::AbstractMatrix, ::Type{BandRange}, j::Integer) = A[colrange(A, j), j]

# scalar - integer - BandRange -- A[1, BandRange]
@inline getindex(A::AbstractMatrix, k::Integer, ::Type{BandRange}) = A[k, rowrange(A, k)]



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
@inline function inbands_setindex!(data::AbstractMatrix{T}, u::Integer, v, k::Integer, j::Integer) where {T}
    data[u + k - j + 1, j] = convert(T, v)::T
    v
end

# slow fall back method
@inline function inbands_setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    # it takes a bit of time to extract A.data, A.u since julia checks if those fields exist
    # the @inbounds here will suppress those checks
    @inbounds r = inbands_setindex!(A.data, A.u, v, k, j)
    r
end

@inline function banded_setindex!(data::AbstractMatrix, l::Int, u::Int, v, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_setindex!(data, u, v, k, j)
    elseif v ≠ 0  # allow setting outside bands to zero
        throw(BandError(data,j-k))
    else # v == 0
        v
    end
end


# scalar - integer - integer
@inline function setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    @boundscheck checkbounds(A, k, j)
    @inbounds r = banded_setindex!(A.data, A.l, A.u, v, k ,j)
    r
end

# scalar - colon - colon
function setindex!(A::BandedMatrix{T}, v, ::Colon, ::Colon) where {T}
    if v == zero(T)
        A.data[:] = convert(T, v)::T
    else
        throw(BandError(A, A.u+1))
    end
end

# scalar - colon
function setindex!(A::BandedMatrix{T}, v, ::Colon) where {T}
    if v == zero(T)
        A.data[:] = convert(T, v)::T
    else
        throw(BandError(A, A.u+1))
    end
end

# matrix - colon - colon
@inline function setindex!(A::BandedMatrix{T}, v::AbstractMatrix, kr::Colon, jr::Colon) where {T}
    @boundscheck checkdimensions(size(A), size(v))
    @boundscheck checkbandmatch(A, v, kr, jr)

    for j=1:size(A,2), k=colrange(A,j)
        @inbounds A[k,j] = v[k,j]
    end
    A
end

function setindex!(A::BandedMatrix{T}, v::AbstractVector, ::Colon) where {T}
    A[:, :] = reshape(v,size(A))
end


# ~ indexing along a band

# scalar - band - colon
@inline function setindex!(A::BandedMatrix{T}, v, b::Band) where {T}
    @boundscheck checkband(A, b)
    A.data[A.u - b.i + 1, :] = convert(T, v)::T
end

# vector - band - colon
@inline function setindex!(A::BandedMatrix{T}, V::AbstractVector, b::Band) where {T}
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

# scalar - colon - integer -- A[:, 1] = 2 - not allowed
function setindex!(A::BandedMatrix{T}, v, kr::Colon, j::Integer) where {T}
    if v == zero(T)
        A.data[:,j] = convert(T, v)::T
    else
        throw(BandError(A, _firstdiagcol(A, j)))
    end
end


# vector - colon - integer -- A[:, 1] = [1, 2, 3] - not allowed
@inline function setindex!(A::BandedMatrix{T}, V::AbstractVector, kr::Colon, j::Integer) where {T}
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(1:size(A,1), V)
    @boundscheck checkbandmatch(A,V,:,j)

    A.data[data_colrange(A,j)] = V[colrange(A,j)]
    V
end

# scalar - BandRange - integer -- A[1, BandRange] = 2
setindex!(A::BandedMatrix{T}, v, ::Type{BandRange}, j::Integer) where {T} =
    (A[colrange(A, j), j] = convert(T, v)::T) # call range method

# vector - BandRange - integer -- A[1, BandRange] = 2
setindex!(A::BandedMatrix, V::AbstractVector, ::Type{BandRange}, j::Integer) =
    (A[colrange(A, j), j] = V) # call range method

# scalar - range - integer -- A[1:2, 1] = 2
@inline function setindex!(A::BandedMatrix, v, kr::AbstractRange, j::Integer)
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
@inline function setindex!(A::BandedMatrix, V::AbstractVector, kr::AbstractRange, j::Integer)
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
function setindex!(A::BandedMatrix{T}, v, k::Integer, jr::Colon) where {T}
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
@inline function setindex!(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Colon) where {T}
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

# scalar - integer - BandRange -- A[1, BandRange] = 2
setindex!(A::BandedMatrix{T}, v, k::Integer, ::Type{BandRange}) where {T} =
    (A[k, rowrange(A, k)] = convert(T, v)::T) # call range method

# vector - integer - BandRange -- A[1, BandRange] = [1, 2, 3]
setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, ::Type{BandRange}) =
    (A[k, rowstart(A, k):rowstop(A, k)] = V) # call range method

# scalar - integer - range -- A[1, 2:3] = 3
@inline function setindex!(A::BandedMatrix{T}, v, k::Integer, jr::AbstractRange) where {T}
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
@inline function setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, jr::AbstractRange)
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
@inline function setindex!(A::BandedMatrix, v, kr::AbstractRange, jr::AbstractRange)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkband(A, kr, jr)
    data, u = A.data, A.u
    for j in jr, k in kr
        inbands_setindex!(data, u, v, k, j)
    end
    v
end

# matrix - range - range
@inline function setindex!(A::BandedMatrix, V::AbstractMatrix, kr::AbstractRange, jr::AbstractRange)
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

# scalar - BandRange -- A[BandRange] = 2
setindex!(A::BandedMatrix{T}, v, ::Type{BandRange}) where {T} =
    A.data[:] = convert(T, v)::T

# ~~ end setindex! ~~



function Base.convert(::Type{Matrix}, A::BandedMatrix)
    ret=zeros(eltype(A),size(A,1),size(A,2))
    for j = 1:size(ret,2), k = colrange(ret,j)
        @inbounds ret[k,j] = A[k,j]
    end
    ret
end

Base.full(A::BandedMatrix) = convert(Matrix, A)


function sparse(B::BandedMatrix)
    i=Vector{Int}(undef,length(B.data)); j=Vector{Int}(undef,length(B.data))
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




# pass standard routines to Matrix

norm(B::BandedMatrix,opts...) = norm(Matrix(B),opts...)


# We turn off bound checking to allow nicer syntax without branching
#setindex!(A::BandedMatrix,v,k::Integer,j::Integer)=((A.l≤j-k≤A.u)&&k≤A.n)?ussetindex!(A,v,k,j):throw(BoundsError())
#setindex!(A::BandedMatrix,v,kr::AbstractRange,j::Integer)=(A.l≤j-kr[end]≤j-kr[1]≤A.u&&kr[end]≤A.n)?ussetindex!(A,v,kr,j):throw(BoundsError())


## ALgebra and other functions

function fill!(A::BandedMatrix{T}, x) where T
    x == zero(T) || throw(BandError(A))
    fill!(A.data, x)
    A
end

if VERSION < v"0.7-"
    function Base.scale!(α::Number, A::BandedMatrix)
        Base.scale!(α, A.data)
        A
    end

    function Base.scale!(A::BandedMatrix, α::Number)
        Base.scale!(A.data, α)
        A
    end
else
    function LinearAlgebra.lmul!(α::Number, A::BandedMatrix)
        LinearAlgebra.lmul!(α, A.data)
        A
    end

    function LinearAlgebra.rmul!(A::BandedMatrix, α::Number)
        LinearAlgebra.rmul!(A.data, α)
        A
    end
end

function Base.transpose(B::BandedMatrix)
    Bt = BandedMatrix(Zeros{eltype(B)}(size(B,2),size(B,1)), (B.u,B.l))
    for j = 1:size(B,2), k = colrange(B,j)
       Bt[j,k]=B[k,j]
    end
    Bt
end

function Base.ctranspose(B::BandedMatrix)
    Bt=BandedMatrix(Zeros{eltype(B)}(size(B,2),size(B,1)), (B.u,B.l))
    for j = 1:size(B,2), k = colrange(B,j)
       Bt[j,k]=conj(B[k,j])
    end
    Bt
end



function diag(A::BandedMatrix{T}) where {T}
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
        $OP(A::BandedMatrix, b::Number) =
            _BandedMatrix($OP(A.data,b),A.m,A.l,A.u)
        broadcast(::typeof($OP), A::BandedMatrix, b::Number) =
            _BandedMatrix($OP.(A.data,b),A.m,A.l,A.u)
    end
end

for OP in (:*,:\)
    @eval begin
        $OP(a::Number, B::BandedMatrix) =
            _BandedMatrix($OP(a,B.data),B.m,B.l,B.u)
        broadcast(::typeof($OP), a::Number, B::BandedMatrix) =
            _BandedMatrix($OP.(a,B.data),B.m,B.l,B.u)
    end
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


## BandedSubBandedMatrix routines
# gives the band which is diagonal for the parent
bandshift(a::AbstractRange,b::AbstractRange) = first(a)-first(b)
bandshift(::Base.Slice{Base.OneTo{Int}},b::AbstractRange) = 1-first(b)
bandshift(a::AbstractRange,::Base.Slice{Base.OneTo{Int}}) = first(a)-1
bandshift(::Base.Slice{Base.OneTo{Int}},b::Base.Slice{Base.OneTo{Int}}) = 0
bandshift(S) = bandshift(parentindices(S)[1],parentindices(S)[2])

bandwidth(S::BandedSubBandedMatrix{T}, k::Integer) where {T} = bandwidth(parent(S),k) + (k==1 ? -1 : 1)*bandshift(S)

@inline function inbands_getindex(S::BandedSubBandedMatrix{T}, k::Integer, j::Integer) where {T}
    @inbounds r = inbands_getindex(parent(S), reindex(S, parentindices(S), (k, j))...)
    r
end

@inline function inbands_setindex!(S::BandedSubBandedMatrix{T}, v, k::Integer, j::Integer) where {T}
    @inbounds r = inbands_setindex!(parent(S), v, reindex(S, parentindices(S), (k, j))...)
    r
end


function Base.convert(::Type{BandedMatrix}, S::BandedSubBandedMatrix{T}) where {T}
    A=parent(S)
    kr,jr=parentindices(S)
    shft=kr[1]-jr[1]
    l,u=bandwidths(A)
    if -u ≤ shft ≤ l
        _BandedMatrix(A.data[:,jr],length(kr),l-shft,u+shft)
    elseif shft > l
        # need to add extra zeros at top since negative bandwidths not supported
        # new bandwidths = (0,u+shft)
        dat = zeros(T,u+shft+1,length(jr))
        dat[1:l+u+1,:] = A.data[:,jr]
        _BandedMatrix(dat,length(kr),0,u+shft)
    else  # shft < -u
        dat = zeros(T,l-shft+1,length(jr))
        dat[-shft-u+1:end,:] = A.data[:,jr]  # l-shft+1 - (-shft-u) == l+u+1
        _BandedMatrix(dat,length(kr),l-shft,0)
    end
end

## These routines give access to the necessary information to call BLAS

@inline leadingdimension(B::BandedMatrix) = stride(B.data,2)
@inline leadingdimension(B::BandedSubBandedMatrix{T}) where {T} = leadingdimension(parent(B))


@inline Base.pointer(B::BandedMatrix) = pointer(B.data)
@inline Base.pointer(B::BandedSubBandedMatrix{T}) where {T} =
    pointer(parent(B))+leadingdimension(parent(B))*(first(parentindices(B)[2])-1)*sizeof(T)
