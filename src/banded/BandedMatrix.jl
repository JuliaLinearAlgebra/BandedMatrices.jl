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

function _BandedMatrix end

struct BandedMatrix{T, CONTAINER, RAXIS} <: AbstractBandedMatrix{T}
    data::CONTAINER  # l+u+1 x n (# of columns)
    raxis::RAXIS # axis for rows (col axis comes from data)
    l::Int # lower bandwidth (possibly negative)
    u::Int # upper bandwidth (possibly negative)
    global function _BandedMatrix(data::AbstractMatrix{T}, raxis::AbstractUnitRange, l, u) where {T}
        if size(data,1) ≠ l+u+1  && !(size(data,1) == 0 && -l > u)
           error("Data matrix must have number rows equal to number of bands")
        else
            new{T, typeof(data), typeof(raxis)}(data, raxis, l, u)
        end
    end
end

_BandedMatrix(data::AbstractMatrix, m::Integer, l, u) = _BandedMatrix(data, oneto(m), l, u)

Base.parent(B::BandedMatrix) = B.data
Base.dataids(B::BandedMatrix) = (Base.dataids(B.data)..., Base.dataids(B.raxis)...)

const DefaultBandedMatrix{T} = BandedMatrix{T,Matrix{T},OneTo{Int}}

bandedcolumns(_) = BandedColumns{UnknownLayout}()
bandedcolumns(::ML) where ML<:AbstractStridedLayout = BandedColumns{ML}()
bandedcolumns(::ML) where ML<:AbstractFillLayout = BandedColumns{ML}()

MemoryLayout(A::Type{BandedMatrix{T,Cont,Axes}}) where {T,Cont,Axes} = bandedcolumns(MemoryLayout(Cont))


## Constructors

"""
    BandedMatrix{T}(undef, (n, m), (l, u))

returns an uninitialized `n`×`m` banded matrix of type `T` with bandwidths `(l,u)`.
"""
BandedMatrix{T, C, OneTo{Int}}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer}) where {T<:BlasFloat, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m), n, a, b)
BandedMatrix{T, C, OneTo{Int}}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer}) where {T, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m),n,a,b)
BandedMatrix{T, C, OneTo{Int}}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer})  where {T<:Number, C<:AbstractMatrix{T}} =
    _BandedMatrix(fill!(similar(C, max(0,b+a+1),m), zero(T)),n,a,b)

BandedMatrix{T, C}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer}) where {T<:BlasFloat, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m), n, a, b)
BandedMatrix{T, C}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer}) where {T, C<:AbstractMatrix{T}} =
    _BandedMatrix(C(undef,max(0,b+a+1),m),n,a,b)
BandedMatrix{T, C}(::UndefInitializer, (n,m)::NTuple{2,Integer}, (a,b)::NTuple{2,Integer})  where {T<:Number, C<:AbstractMatrix{T}} =
    _BandedMatrix(fill!(convert(C, similar(C, max(0,b+a+1),m)), zero(T)),n,a,b)

BandedMatrix{T}(::UndefInitializer, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}) where T =
BandedMatrix{T, Matrix{T}}(undef,nm,ab)
BandedMatrix{T}(::UndefInitializer, nm::NTuple{2,OneTo{Int}}, ab::NTuple{2,Integer}) where T =
    BandedMatrix{T}(undef, length.(nm), ab)

@deprecate BandedMatrix{T, C}(::UndefInitializer, n::Integer, ::Colon, a::Integer, b::Integer)  where {T, C<:AbstractMatrix{T}} BandedMatrix{T,C}(undef,n,n+b,a,b)
@deprecate BandedMatrix{T, C}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer) where {T, C<:AbstractMatrix{T}} BandedMatrix{T, C}(undef, (n,m), (a,b))
@deprecate BandedMatrix{T}(::UndefInitializer, n::Integer, m::Integer, a::Integer, b::Integer)  where {T} BandedMatrix{T}(undef,(n,m),(a,b))
@deprecate BandedMatrix{T}(::UndefInitializer, n::Integer, ::Colon, a::Integer, b::Integer)  where {T} BandedMatrix{T}(undef,(n,n+b),(a,b))

BandedMatrix{V}(M::BandedMatrix) where {V} = _BandedMatrix(AbstractMatrix{V}(M.data), M.raxis, M.l, M.u)
BandedMatrix(M::BandedMatrix{V}) where {V} = _BandedMatrix(AbstractMatrix{V}(M.data), M.raxis, M.l, M.u)

function BandedMatrix{T, C, Ax}(A::UniformScaling, nm::NTuple{2,Integer}, (l,u)::NTuple{2,Integer}=(0,0)) where {T,C,Ax}
    ret = BandedMatrix{T, C, Ax}(undef, nm, (l,u))
    zero!(ret.data)
    ret.data[u+1,:] .= A.λ
    ret
end
BandedMatrix{T,C}(A::UniformScaling, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}=(0,0)) where {T,C} = 
    BandedMatrix{T, C, OneTo{Int}}(A, nm, ab)
BandedMatrix{T}(A::UniformScaling, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}=(0,0)) where T = 
    BandedMatrix{T, Matrix{T}}(A, nm, ab)
BandedMatrix(A::UniformScaling, nm::NTuple{2,Integer}, ab::NTuple{2,Integer}=(0,0)) = 
    BandedMatrix{eltype(A)}(A, nm, ab)


convert(::Type{BandedMatrix{V}}, M::BandedMatrix{V}) where {V} = M
convert(::Type{BandedMatrix{V}}, M::BandedMatrix) where {V} =
        _BandedMatrix(convert(AbstractMatrix{V}, M.data), M.raxis, M.l, M.u)
convert(::Type{BandedMatrix}, M::BandedMatrix) = M

convert(BM::Type{BandedMatrix{T,C}}, M::BandedMatrix{T,C}) where {T,C<:AbstractMatrix{T}} = M
convert(BM::Type{BandedMatrix{T,C,AXIS}}, M::BandedMatrix{T,C,AXIS}) where {T,C<:AbstractMatrix{T},AXIS} = M
convert(BM::Type{BandedMatrix{T,C,OneTo{Int}}}, M::BandedMatrix{T,C,OneTo{Int}}) where {T,C<:AbstractMatrix{T}} = M

function convert(::Type{BandedMatrix{<:, C}}, M::BandedMatrix) where C
    M.data isa C && return M
    _BandedMatrix(convert(C, M.data), M.raxis, M.l, M.u)
end
function convert(BM::Type{BandedMatrix{T, C}}, M::BandedMatrix) where {T, C <: AbstractMatrix{T}}
    M.data isa C && return M
    _BandedMatrix(convert(C, M.data), M.raxis, M.l, M.u)
end
convert(BM::Type{BandedMatrix{T, C, AXIS}}, M::BandedMatrix) where {T, C <: AbstractMatrix{T}, AXIS} =
    _BandedMatrix(convert(C, M.data), convert(AXIS, M.raxis), M.l, M.u)
convert(BM::Type{BandedMatrix{T, C, OneTo{Int}}}, M::BandedMatrix) where {T, C <: AbstractMatrix{T}} =
    _BandedMatrix(convert(C, M.data), convert(OneTo{Int}, M.raxis), M.l, M.u)

for MAT in (:AbstractBandedMatrix, :AbstractMatrix, :AbstractArray)
    @eval begin
        convert(::Type{$MAT{T}}, M::BandedMatrix) where {T} = convert(BandedMatrix{T}, M)
        convert(::Type{$MAT{T}}, M::BandedMatrix{T}) where {T} = convert(BandedMatrix{T}, M)
        convert(::Type{$MAT}, M::BandedMatrix) = M
        $MAT{T}(M::BandedMatrix) where {T} = BandedMatrix{T}(M)
        $MAT(M::BandedMatrix{T}) where {T} = BandedMatrix{T}(M)
    end
end

function copyto_bandeddata!(B::BandedMatrix, M::AbstractMatrix)
    copyto_bandeddata!(B, MemoryLayout(B), M, MemoryLayout(M))
end
function copyto_bandeddata!(B, ::BandedColumns{DenseColumnMajor},
        M, ::Union{BandedColumns{DenseColumnMajor}, DiagonalLayout{DenseColumnMajor}})
    copyto!(B.data, bandeddata(M))
    return B
end
copyto_bandeddata!(B, @nospecialize(_), M, @nospecialize(_)) = copyto!(B, M)

@inline function _convert_common(T, Container, M)
    B = BandedMatrix{T, Container}(undef, size(M), bandwidths(M))
    copyto_bandeddata!(B, M)
end

function convert(::Type{BandedMatrix{<:,C,OneTo{Int}}}, M::AbstractMatrix) where {C}
    Container = typeof(convert(C, similar(M, 0, 0)))
    T = eltype(Container)
    _convert_common(T, Container, M)
end

@inline function _convert_common_container(T, C, M)
    Container = typeof(convert(C, similar(M, T, 0, 0)))
    _convert_common(T, Container, M)
end

function convert(::Type{BandedMatrix{T,C,OneTo{Int}}}, M::AbstractMatrix) where {T, C}
    _convert_common_container(T, C, M)
end

function convert(::Type{BandedMatrix{T,C,OneTo{Int}}}, M::AbstractMatrix) where {T, C<:AbstractMatrix{T}}
    _convert_common_container(T, C, M)
end

convert(::Type{BandedMatrix{<:, C}}, M::AbstractMatrix) where {C} = convert(BandedMatrix{<:,C,OneTo{Int}}, M)
convert(::Type{BandedMatrix{T,C}}, M::AbstractMatrix) where {T, C} = convert(BandedMatrix{T,C,OneTo{Int}}, M)

convert(::Type{BandedMatrix{T}}, M::AbstractMatrix) where {T} =
    convert(BandedMatrix{T, typeof(similar(M, T, 0, 0))}, M)

convert(::Type{BandedMatrix}, M::AbstractMatrix) = convert(BandedMatrix{eltype(M)}, M)
convert(::Type{DefaultBandedMatrix}, M::AbstractMatrix{T}) where T = convert(DefaultBandedMatrix{T}, M)

copy(B::BandedMatrix) = _BandedMatrix(copy(B.data), B.raxis, B.l, B.u)


copymutable_oftype_layout(::BandedColumns, B, ::Type{S}) where S =
    _BandedMatrix(LinearAlgebra.copymutable_oftype(bandeddata(B), S), axes(B,1), bandwidths(B)...)

copymutable_oftype_layout(::BandedRows, B, ::Type{S}) where S =
    dualadjoint(B)(LinearAlgebra.copymutable_oftype(parent(B), S))

copymutable_oftype_layout(::AbstractBandedLayout, B, ::Type{S}) where S =
    copyto!(BandedMatrix{S}(undef, axes(B), bandwidths(B)), B)


promote_rule(::Type{BandedMatrix{T1, C1}}, ::Type{BandedMatrix{T2, C2}}) where {T1,C1, T2,C2} =
    BandedMatrix{promote_type(T1,T2), promote_type(C1, C2)}


for (op,bop) in ((:rand,:brand), (:randn,:brandn))
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

"""
    brandn(T,n,m,l,u)

Creates an `n×m` banded matrix  with random normals in the bandwidth of type `T` with bandwidths `(l,u)`
"""
brandn

## Conversions from AbstractArrays, we include FillArrays in case `zeros` is ever faster
BandedMatrix{T}(A::AbstractMatrix, bnds::NTuple{2,Integer}) where T =
    BandedMatrix{T, Matrix{T}}(A, bnds)
function BandedMatrix{T, C}(A::AbstractMatrix, bnds::NTuple{2,Integer}) where {T, C <: AbstractMatrix{T}}
    (n,m) = size(A)
    (l,u) = bnds
    ret = BandedMatrix{T, C}(undef, (n, m), (l, u))
    @inbounds for j = 1:m, k = max(1,j-u):min(n,j+l)
        inbands_setindex!(ret, A[k,j], k, j)
    end
    ret
end

BandedMatrix(A::AbstractMatrix{T}, bnds::NTuple{2,Integer}) where T =
    BandedMatrix{T}(A, bnds)


BandedMatrix{V,C}(Z::Zeros{T,2}, bnds::NTuple{2,Integer}) where {T,C,V} =
    _BandedMatrix(C(Zeros{T}(max(0,sum(bnds)+1),size(Z,2))),size(Z,1),bnds...)

BandedMatrix{V,C,Base.OneTo{Int}}(Z::Zeros{T,2}, bnds::NTuple{2,Integer}) where {T,C,V} =
    _BandedMatrix(C(Zeros{T}(max(0,sum(bnds)+1),size(Z,2))),size(Z,1),bnds...)

BandedMatrix{V}(Z::Zeros{T,2}, bnds::NTuple{2,Integer}) where {T,V} =
    _BandedMatrix(zeros(V,max(0,sum(bnds)+1),size(Z,2)),size(Z,1),bnds...)

BandedMatrix(E::RectDiagonal{T}, bnds::NTuple{2,Integer}) where T = BandedMatrix{T}(E, bnds)
function BandedMatrix{T}(E::RectDiagonal, bnds::NTuple{2,Integer}) where T
    ret = BandedMatrix(Zeros{T}(E), bnds)
    ret[band(0)] .= E.diag
    ret
end

BandedMatrix{T,C}(A::AbstractMatrix) where {T, C<:AbstractMatrix{T}} =
    copyto_bandeddata!(BandedMatrix{T, C}(undef, size(A), bandwidths(A)), A)
BandedMatrix{T}(A::AbstractMatrix) where T =
    copyto_bandeddata!(BandedMatrix{T}(undef, size(A), bandwidths(A)), A)


_BandedMatrix(_, A::AbstractMatrix) = BandedMatrix(A, bandwidths(A))
BandedMatrix(A::AbstractMatrix) = _BandedMatrix(MemoryLayout(A), A)


## specialised
# use bandeddata if possible
_BandedMatrix(::BandedColumns, A::AbstractMatrix) = _BandedMatrix(copy(bandeddata(A)), axes(A,1), bandwidths(A)...)
function _BandedMatrix(::BandedRows, A::AbstractMatrix)
    bdata = bandedrowsdata(A)
    data = similar(bdata, eltype(bdata), reverse(size(bdata)))
    u, ℓ = bandwidths(A)
    n = size(A, 2)
    for j in axes(A, 1) # Construct the data for A by flipping bands
        for i in max(1, j - u):min(n, j + ℓ)
            data[ℓ + 1 + j - i, i] = bdata[j, u+1+i-j]
        end
    end
    return _BandedMatrix(data, axes(A, 1), bandwidths(A)...)
end
function _BandedMatrix(::DiagonalLayout, A::AbstractMatrix{T}) where T
    m,n = size(A)
    dat = Matrix{T}(undef, 1, n)
    copyto!(vec(dat), diagonaldata(A))
    _BandedMatrix(dat, m, 0, 0)
end
function _BandedMatrix(::BidiagonalLayout, A::AbstractMatrix{T}) where T
    m,n = size(A)
    dat = Matrix{T}(undef, 2, n)
    if bidiagonaluplo(A) == 'L'
        copyto!(view(dat, 1, :), diagonaldata(A))
        copyto!(view(dat, 2, 1:n-1), subdiagonaldata(A))
        _BandedMatrix(dat, m, 1, 0)
    else # u
        copyto!(view(dat, 1, 2:n), supdiagonaldata(A))
        copyto!(view(dat, 2, :), diagonaldata(A))
        _BandedMatrix(dat, m, 0, 1)
    end
end

function _BandedMatrix(::AbstractTridiagonalLayout, A::AbstractMatrix{T}) where T
    m,n = size(A)
    dat = Matrix{T}(undef, 3, n)

    copyto!(view(dat, 1, 2:n), supdiagonaldata(A))
    copyto!(view(dat, 2, :), diagonaldata(A))
    copyto!(view(dat, 3, 1:n-1), subdiagonaldata(A))
    _BandedMatrix(dat, m, 1, 1)
end


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

similar(bm::AbstractBandedMatrix, n::Integer, m::Integer) = similar(bm, eltype(bm), n, m)
similar(bm::AbstractBandedMatrix, n::Integer, m::Integer, l::Integer, u::Integer) =
    similar(bm, eltype(bm), m, n, l, u)
similar(bm::AbstractBandedMatrix, nm::Tuple{<:Integer,<:Integer}) = similar(bm, nm...)




## Abstract Array Interface

axes(A::BandedMatrix) = (A.raxis, axes(A.data,2))
size(A::BandedMatrix) = (length(A.raxis), size(A.data,2))


## banded matrix interface
bandeddata(A::BandedMatrix) = A.data
bandwidths(A::BandedMatrix) = (A.l , A.u)

# ~~ getindex ~~

# fast method used below
@propagate_inbounds inbands_getindex(data::AbstractMatrix, u::Integer, k::Integer, j::Integer) =
    data[u + k - j + 1, j]

@propagate_inbounds function inbands_getindex(A::BandedMatrix, k::Integer, j::Integer)
    inbands_getindex(A.data, A.u, k, j)
end

# work around for Any matrices
_offband_zero(::AbstractMatrix{T}, _, _, _, _) where T = zero(T)
_offband_zero(::AbstractMatrix{Any}, _, _, _, _) = nothing
_offband_zero(data::AbstractMatrix{<:AbstractMatrix}, l, u, k, j) =
    diagzero(Diagonal(view(data,u+1,:)), k, j)
diagzero(D::Diagonal{B}, i, j) where B<:BandedMatrix =
    B(Zeros{eltype(B)}(size(D.diag[i], 1), size(D.diag[j], 2)), (bandwidth(D.diag[i],1), bandwidth(D.diag[j],2)))

# banded get index, used for banded matrices with other data types
@propagate_inbounds function banded_getindex(data::AbstractMatrix, l::Integer, u::Integer, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_getindex(data, u, k, j)
    else
        _offband_zero(data, l, u, k, j)
    end
end


# Int - Int
@inline function getindex(A::BandedMatrix, k::Int, j::Int)
    @boundscheck checkbounds(A, k, j)
    @inbounds r = banded_getindex(A.data, A.l, A.u, k, j)
    r
end

# BandRange - Int
@propagate_inbounds function getindex(A::BandedMatrix, ::BandRangeType, j::Int)
    @boundscheck checkbounds(A, colrange(A, j), j)
    A.data[data_colrange(A,j)]
end

# Colon - Int
@propagate_inbounds function getindex(A::BandedMatrix, ::Colon, j::Int)
    @boundscheck checkbounds(A, axes(A,1), j)
    r = similar(A, axes(A,1))
    r[firstindex(r):min(size(A, 1), colstart(A,j)-1)] .= zero(eltype(r))
    # broadcasted assignment is currently faster than setindex
    # see https://github.com/JuliaLang/julia/issues/40962#issuecomment-1921340377
    # may need revisiting in the future
    r[colrange(A,j)] .= @view A.data[data_colrange(A,j)]
    r[colstop(A,j)+1:end] .= zero(eltype(r))
    return r
end

# Int - BandRange
@propagate_inbounds function getindex(A::BandedMatrix, k::Int, j::BandRangeType)
    @boundscheck checkbounds(A, k, rowrange(A, k))
    A.data[data_rowrange(A,k)]
end

# Int - Colon
@propagate_inbounds function getindex(A::BandedMatrix, k::Int, ::Colon)
    @boundscheck checkbounds(A, k, axes(A,2))
    r = similar(A, axes(A,2))
    r[firstindex(r):min(size(A, 2), rowstart(A,k)-1)] .= zero(eltype(r))
    r[rowrange(A,k)] = @view A.data[data_rowrange(A,k)]
    r[rowstop(A,k)+1:end] .= zero(eltype(r))
    return r
end

# ~ indexing along a band
# we reduce it to converting a View


"""
    BandedMatrixBand

Type to represent a view of a band of a `BandedMatrix`

# Examples
```jldoctest
julia> B = BandedMatrix(0=>1:3);

julia> view(B, band(0)) isa BandedMatrices.BandedMatrixBand
true
```
"""
const BandedMatrixBand{T} = Union{
                                SubArray{T, 1, <:ReshapedArray{T,1,
                                    <:Union{BandedMatrix{T}, SubArray{T,2,<:BandedMatrix{T},
                                    <:NTuple{2, AbstractUnitRange{Int}}}}}, <:Tuple{BandSlice{<:Integer}}},
                                SubArray{T, 1, <:BandedMatrix{T}, <:Tuple{BandSlice{<:CartesianIndex{2}}}}}


band(V::BandedMatrixBand) = first(parentindices(V)).band.i


"""
    dataview(V::BandedMatrices.BandedMatrixBand)

Forward a view of a band of a `BandedMatrix` to the parent's data matrix.

!!! warn
    This will error if the indexing is out-of-bounds for the data matrix, even if it is inbounds
    for the parent `BandedMatrix`

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7, -1=>8:10)
4×4 BandedMatrix{Int64} with bandwidths (1, 1):
 1  5   ⋅  ⋅
 8  2   6  ⋅
 ⋅  9   3  7
 ⋅  ⋅  10  4

julia> v = view(A, band(1));

julia> BandedMatrices.dataview(v)
3-element view(::Matrix{Int64}, 1, 2:4) with eltype Int64:
 5
 6
 7
```
"""
function dataview(V::BandedMatrixBand)
    A = parent(parent(V))
    b = band(V)
    m,n = size(A)
    u = bandwidth(A,2)
    view(bandeddata(A), u - b + 1, max(b,0)+1:min(n,m+b))
end

@propagate_inbounds function getindex(B::BandedMatrixBand, i::Int)
    A = parent(parent(B))
    b = band(B)
    l, u = bandwidths(A)
    if -l ≤ band(B) ≤ u
        dataview(B)[i]
    else
        @boundscheck checkbounds(B, i)
        zero(eltype(B))
    end
end

# B[band(i)]
@inline function copyto!(v::AbstractVector, B::BandedMatrixBand)
    A = parent(parent(B))
    l, u = bandwidths(A)
    if -l ≤ band(B) ≤ u
        copyto!(v, dataview(B))
    else
        Binds = axes(B,1)
        v[firstindex(v)-1 .+ Binds] .= 0
    end
    return v
end

# B[band(i)] .= x::Number
@inline function fill!(Bv::BandedMatrixBand, x)
    b = band(Bv)
    A = parent(parent(Bv))
    l, u = bandwidths(A)
    if -l <= b <= u
        fill!(dataview(Bv), x)
    elseif !iszero(x)  # allow setting outside bands to zero
        throw(BandError(A,b))
    end
    Bv
end

@noinline throwdm(destaxes, srcaxes) =
    throw(DimensionMismatch("destination axes $destaxes do not match source axes $srcaxes"))

# more complicated broadcating
# e.g. B[band(i)] .= a .* x .+ v
@inline function copyto!(dest::BandedMatrixBand, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(src))

    A = parent(parent(dest))
    if -A.l ≤ band(dest) ≤ A.u
        copyto!(dataview(dest), bc)
    else
        all(iszero, bc) || throw(BandError(A, band(dest)))
    end
    return dest
end

# ~ indexing along a row



# give range of data matrix corresponding to colrange/rowrange
data_colrange(A::BandedMatrix{T}, i::Integer) where {T} =
    (max(1,A.u+2-i):min(size(A,1)+A.u-i+1,size(A.data,1))) .+
                                ((i-1)*size(A.data,1))

function data_rowrange(A::BandedMatrix, rowind::Integer)
    StepRangeLen(rowind ≤ 1+A.l ? A.u+rowind : (rowind-A.l)*size(A.data,1),
        size(A.data,1)-1,
        length(rowrange(A,rowind))
    )
end

# ~~ setindex! ~~

# ~ Special setindex methods ~

# fast method used below
@propagate_inbounds function inbands_setindex!(data::AbstractMatrix, u::Integer, v, k::Integer, j::Integer)
    data[u + k - j + 1, j] = v
end

# slow fall back method
@propagate_inbounds function inbands_setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    inbands_setindex!(A.data, A.u, v, k, j)
end

@propagate_inbounds function banded_setindex!(data::AbstractMatrix, l::Integer, u::Integer, v, k::Integer, j::Integer)
    if -l ≤ j-k ≤ u
        inbands_setindex!(data, u, v, k, j)
    elseif !iszero(v)  # allow setting outside bands to zero
        throw(BandError(data,j-k))
    else # v == 0
        v
    end
end

# scalar - integer - integer
@inline function setindex!(A::BandedMatrix, v, k::Integer, j::Integer)
    @boundscheck checkbounds(A, k, j)
    @inbounds banded_setindex!(A.data, A.l, A.u, v, k ,j)
    A
end

# matrix - colon - colon
@inline function setindex!(A::BandedMatrix{T}, v::AbstractMatrix, kr::Colon, jr::Colon) where {T}
    @boundscheck checkdimensions(size(A), size(v))
    @boundscheck checkbandmatch(A, v, kr, jr)

    data = A.data
    l,u = bandwidths(A)

    for j=rowsupport(A), k=colrange(A,j)
        @inbounds inbands_setindex!(data, u, v[k,j], k, j)
    end
    A
end

@propagate_inbounds function setindex!(A::BandedMatrix{T}, v::AbstractVector, ::Colon) where {T}
    A[:, :] = reshape(v,size(A))
    A
end


# ~ indexing along a band

# vector - band - colon
@inline function setindex!(A::BandedMatrix{T}, V::AbstractVector, b::Band) where {T}
    @boundscheck checkband(A, b)
    @boundscheck checkdimensions(diaglength(A, b), V)
    row = A.u - b.i + 1
    data, i = A.data, max(b.i + 1, 1)
    for v in V
        @inbounds data[row, i] = v
        i += 1
    end
    A
end


# ~ indexing along columns
# vector - colon - integer -- A[:, 1] = [1, 2, 3] - not allowed
@propagate_inbounds function setindex!(A::BandedMatrix{T}, V::AbstractVector, kr::Colon, j::Integer) where {T}
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(1:size(A,1), V)
    @boundscheck checkbandmatch(A,V,:,j)

    A.data[data_colrange(A,j)] = @view V[colrange(A,j)]
    A
end

# vector - BandRange - integer -- A[BandRange, 1] = [1, 2, 3]
@propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, ::BandRangeType, j::Integer)
    kr = colrange(A, j)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(kr, V)

    data, u = A.data, A.u
    for i in eachindex(kr, V)
        inbands_setindex!(data, u, V[i], kr[i], j)
    end
    A
end

# vector - range - integer -- A[1:3, 1] = [1, 2, 3]
@propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, kr::AbstractRange, j::Integer)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(kr, V)
    @boundscheck checkbandmatch(A, V, kr, j)

    a = colstart(A, j)
    b = colstop(A, j)

    data, u, i = A.data, A.u, 0
    for v in V
        k = kr[i+=1]
        if a ≤ k ≤ b
            inbands_setindex!(data, u, v, k, j)
        end
    end
    A
end


# ~ indexing along a row

# vector - integer - colon -- A[1, :] = [1, 2, 0] -- out-of-band values must be zeros
@propagate_inbounds function setindex!(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Colon) where {T}
    @boundscheck if k < 1 || k > size(A,1)
        throw(BoundsError(A, (k, jr)))
    end
    @boundscheck if size(A,2) ≠ length(V)
        throw(DimensionMismatch("tried to assign $(length(V)) vector to $(size(A,1)) destination"))
    end
    @boundscheck checkbandmatch(A, V, k, jr)

    A.data[data_rowrange(A,k)] = @view V[rowrange(A,k)]
    A
end

# vector - integer - BandRange -- A[1, BandRange] = [1, 2, 3]
@propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, ::BandRangeType)
    jr = rowrange(A, k)
    @boundscheck checkbounds(A, k, jr)
    @boundscheck checkdimensions(jr, V)

    data, u = A.data, A.u
    for i in eachindex(jr, V)
        inbands_setindex!(data, u, V[i], k, jr[i])
    end
    A
end

# vector - integer - range -- A[1, 2:3] = [3, 4]
@propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractVector, k::Integer, jr::AbstractRange)
    @boundscheck checkbounds(A, k, jr)
    @boundscheck checkdimensions(jr, V)
    @boundscheck checkbandmatch(A, V, k, jr)

    a = rowstart(A, k)
    b = rowstop(A, k)

    data, u, i = A.data, A.u, 0
    for v in V
        j = jr[i+=1]
        if a ≤ j ≤ b
            inbands_setindex!(data, u, v, k, j)
        end
    end
    A
end

# ~ indexing over a rectangular block

# matrix - range - range
@propagate_inbounds function setindex!(A::BandedMatrix, V::AbstractMatrix, kr::AbstractRange, jr::AbstractRange)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkdimensions(kr, jr, V)
    @boundscheck checkbandmatch(A, V, kr, jr)
    copyto!(view(A, kr, jr), V)
    A
end

# ~~ end setindex! ~~



function convert(::Type{Matrix}, A::BandedMatrix)
    ret=zeros(eltype(A), size(A))
    for j = rowsupport(A), k = colrange(ret,j)
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

function _bidiagonalize!(A::AbstractMatrix{T}, M::BandedColumnMajor) where T <: Complex
    m, n = size(A)
    mn = min(m, n)
    d = Vector{real(T)}(undef, mn)
    e = Vector{real(T)}(undef, mn-1)
    Q = Matrix{T}(undef, 0, 0)
    Pt = Matrix{T}(undef, 0, 0)
    C = Matrix{T}(undef, 0, 0)
    work = Vector{T}(undef, 2*max(m, n))
    rwork = Vector{real(T)}(undef, 2*max(m, n))
    gbbrd!('N', m, n, 0, bandwidth(A, 1), bandwidth(A, 2), bandeddata(A), d, e, Q, Pt, C, work, rwork)
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

"""
allinbands(A)

returns true if there are no entries outside the bands.
"""
function allinbands(A)
    m,n = size(A)
    l,u = bandwidths(A)
    m ≤ l+1 && n ≤ u+1
end

function fill!(A::BandedMatrix, x)
    iszero(x) || allinbands(A) || throw(BandError(A))
    fill!(A.data, x)
    A
end

function LinearAlgebra.fillband!(A::BandedMatrix{T}, x, l, u) where T
    fill!(view(A.data, max(A.u+1-u,1):min(A.u+1-l,size(A.data,1)), :), x)
    A
end

diag(A::BandedMatrix, k::Integer = 0) = A[band(k)]

## BandedSubBandedMatrix routines

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
sublayout(::AbstractBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = BandedLayout()
sublayout(::BandedColumns{L}, ::Type{<:Tuple{AbstractUnitRange,J}}) where {L,J<:AbstractUnitRange} =
    bandedcolumns(sublayout(L(),Tuple{Slice{OneTo{Int}},J}))
sublayout(::BandedRows{L}, ::Type{<:Tuple{J,AbstractUnitRange}}) where {L,J<:AbstractUnitRange} =
    transposelayout(bandedcolumns(sublayout(L(),Tuple{Slice{OneTo{Int}},J})))

Base.permutedims(A::Symmetric{<:Any,<:AbstractBandedMatrix}) = A
Base.permutedims(A::BandedMatrix{<:Number}) = transpose(A) # temp
Base.permutedims(A::Adjoint{<:Real,<:BandedMatrix}) = A' # temp
Base.permutedims(A::Transpose{<:Number,<:BandedMatrix}) = transpose(A) # temp
Base.permutedims(A::BandedMatrix) = PermutedDimsArray(A, (2,1))
bandwidths(A::PermutedDimsArray{<:Any,2,(2,1),(2,1)}) = reverse(bandwidths(parent(A)))

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

function bandeddata(V::SubArray)
    l,u = bandwidths(V)
    L,U = bandwidths(parent(V)) .+ (-1,1) .* bandshift(V)
    view(bandeddata(parent(V)), U-u+1:U+l+1, parentindices(V)[2])
end

bandwidths(S::SubArray{T,2,<:AbstractMatrix,I}) where {T,I<:Tuple{Vararg{AbstractUnitRange}}} =
    min.(size(S) .- 1, bandwidths(parent(S)) .+ (-1,1) .* bandshift(S))

@propagate_inbounds function inbands_getindex(S::BandedSubBandedMatrix{T}, k::Integer, j::Integer) where {T}
    inbands_getindex(parent(S), reindex(parentindices(S), (k, j))...)
end

@propagate_inbounds function inbands_setindex!(S::BandedSubBandedMatrix{T}, v, k::Integer, j::Integer) where {T}
    inbands_setindex!(parent(S), v, reindex(parentindices(S), (k, j))...)
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

_banded_summary(io, B::BandedMatrix{T}, inds) where T = print(io, Base.dims2string(length.(inds)), " BandedMatrix{$T} with bandwidths $(bandwidths(B))")
Base.array_summary(io::IO, B::DefaultBandedMatrix, inds::Tuple{Vararg{OneTo}}) = _banded_summary(io, B, inds)
function Base.array_summary(io::IO, B::BandedMatrix, inds::Tuple{Vararg{OneTo}})
    _banded_summary(io, B, inds)
    print(io, " with data ")
    summary(io, B.data)
end
function Base.array_summary(io::IO, B::BandedMatrix, inds)
    _banded_summary(io, B, inds)
    print(io, " with data ")
    summary(io, B.data)
    print(io, " with indices ", Base.inds2string(inds))
end

## Broadcast style
# allow special casing


bandedbroadcaststyle(_) = BandedStyle()
BroadcastStyle(::Type{<:BandedMatrix{<:Any,Dat}}) where Dat = bandedbroadcaststyle(BroadcastStyle(Dat))

function banded_axpy!(a::Number, X::BandedMatrix, Y::BandedMatrix)
    bx = bandwidths(X)
    by = bandwidths(Y)
    if bx == by
        axpy!(a, X.data, Y.data)
    else
        banded_generic_axpy!(a, X, Y)
    end
    return Y
end

# show
function show(io::IO, B::BandedMatrix)
    print(io, "BandedMatrix(")
    l, u = bandwidths(B)
    limit = get(io, :limit, true)
    br = limit ? intersect(-l:u, range(-l,length=4)) : -l:u
    for (ind, b) in enumerate(br)
        r = @view B[band(b)]
        show(io, b => r)
        b == last(br) || print(io, ", ")
    end
    if limit && br != -l:u
        br2 = max(maximum(br)+1, u-3):u
        if minimum(br2) == maximum(br)+1
            print(io, ", ")
        else
            print(io, "  …  ")
        end
        for (ind, b) in enumerate(br2)
            r = @view B[band(b)]
            show(io, b => r)
            b == u || print(io, ", ")
        end
    end
    print(io, ")")
end


###
# resize
###

function resize(A::BandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(bandeddata(A)), (l+u+1)*m), l+u+1, m), n, l,u)
end
function resize(A::BandedSubBandedMatrix, n::Integer, m::Integer)
    l,u = bandwidths(A)
    _BandedMatrix(reshape(resize!(vec(copy(bandeddata(A))), (l+u+1)*m), l+u+1, m), n, l,u)
end

###
# one
###

function one(A::BandedMatrix)
    m,n = size(A)
    m==n || throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    typeof(A)(I, (m,n))
end

function Base.unaliascopy(B::BandedMatrix)
    _BandedMatrix(Base.unaliascopy(B.data), Base.unaliascopy(B.raxis), B.l, B.u)
end
