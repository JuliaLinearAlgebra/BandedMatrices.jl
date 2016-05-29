__precompile__()

module BandedMatrices
using Base

import Base: getindex, setindex!, *, .*, +, .+, -, .-, ==, <, <=, >,
                >=, ./, /, .^, ^, \, transpose, showerror


import Base: convert, size

import Base.BLAS: libblas

import Base.LinAlg: BlasInt,
                    BlasReal,
                    BlasFloat,
                    BlasComplex,
                    A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!,
                    copy_oftype

import Base.LAPACK: gbtrs!,
                    gbtrf!

import Base: lufact

export BandedMatrix, 
       bandrange, 
       bzeros,
       beye,
       brand,
       bones,
       bandwidth,
       BandError, 
       band, 
       BandRange


# AbstractBandedMatrix must implement

abstract AbstractBandedMatrix{T} <: AbstractSparseMatrix{T,Int}


bandwidths(A::AbstractBandedMatrix) = bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractBandedMatrix) = -bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractBandedMatrix,k::Integer) = k==1?-bandwidth(A,1):bandwidth(A,2)
bandrange(A::AbstractBandedMatrix) = -bandwidth(A,1):bandwidth(A,2)



function getindex(A::AbstractBandedMatrix,k::Integer,j::Integer)
    if k>size(A,1) || j>size(A,2)
        throw(BoundsError(A,(k,j)))
    elseif (bandinds(A,1)≤j-k≤bandinds(A,2))
        unsafe_getindex(A,k,j)
    else
        zero(eltype(A))
    end
end

# override bandwidth(A,k) for each AbstractBandedMatrix
# override unsafe_getindex(A,k,j)



## Gives an iterator over the banded entries of a BandedMatrix

immutable BandedIterator
    m::Int
    n::Int
    l::Int
    u::Int
end


Base.start(B::BandedIterator)=(1,1)


Base.next(B::BandedIterator,state)=
    state,ifelse(state[1]==min(state[2]+B.l,B.m),
                (max(state[2]+1-B.u,1),state[2]+1),
                (state[1]+1,  state[2])
                 )

Base.done(B::BandedIterator,state)=state[2]>B.n || state[2]>B.m+B.u

Base.eltype(::Type{BandedIterator})=Tuple{Int,Int}

# Commented out since there's an error

# for m x n matrix, assuming m ≥ n
# see notes
function bandslength(m,n,l,u)
    if m ≥ n
        (1 + m - n)*n + (-l - l^2 + m + 2l*m - m^2 - n + n^2)÷2 +
            (-u + 2n*u - u^2)÷2
    else
        n*(l+1)+u*n-(u^2+u)÷2
    end
end


function Base.length(B::BandedIterator)
    if B.m ≥ B.n
        bandslength(B.m,B.n,B.l,B.u)
    else # transpose
        bandslength(B.n,B.m,B.u,B.l)
    end
end

# returns an iterator of each index in the banded part of a matrix
eachbandedindex(B)=BandedIterator(size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))




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
    function BandedMatrix(data::Matrix{T},m,l,u)
        if l < 0 || u < 0
            error("Bandwidths must be non-negative")
        elseif size(data,1)!=l+u+1
            error("Data matrix must have number rows equal to number of bands")
        else
            new(data,m,l,u)
        end
    end
end


include("BandedLU.jl")
include("blas.jl")


## Constructors

BandedMatrix(data::Matrix,m::Integer,a::Integer,b::Integer) = BandedMatrix{eltype(data)}(data,m,a,b)

BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) = BandedMatrix{T}(Array(T,b+a+1,m),n,a,b)
BandedMatrix{T}(::Type{T},n::Integer,a::Integer,b::Integer) = BandedMatrix(T,n,n,a,b)
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) = BandedMatrix(T,n,n+b,a,b)


BandedMatrix(data::Matrix,m::Integer,a) = BandedMatrix(data,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a) = BandedMatrix(T,n,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a) = BandedMatrix(T,n,:,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,a) = BandedMatrix(T,n,-a[1],a[end])


Base.convert{V}(::Type{BandedMatrix{V}},M::BandedMatrix) = BandedMatrix{V}(convert(Matrix{V},M.data),M.m,M.l,M.u)
function Base.convert{BM<:BandedMatrix}(::Type{BM},M::Matrix)
    ret=BandedMatrix(eltype(BM)==Any?eltype(M):promote_type(eltype(BM),eltype(M)),size(M,1),size(M,2),size(M,1)-1,size(M,2)-1)
    for k=1:size(M,1),j=1:size(M,2)
        ret[k,j]=M[k,j]
    end
    ret
end


Base.promote_rule{T,V}(::Type{BandedMatrix{T}},::Type{BandedMatrix{V}})=BandedMatrix{promote_type(T,V)}



for (op,bop) in ((:(Base.rand),:brand),(:(Base.zeros),:bzeros),(:(Base.ones),:bones))
    @eval begin
        $bop{T}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer)=BandedMatrix($op(T,b+a+1,m),n,a,b)
        $bop{T}(::Type{T},n::Integer,a::Integer,b::Integer)=$bop(T,n,n,a,b)
        $bop{T}(::Type{T},n::Integer,::Colon,a::Integer,b::Integer)=$bop(T,n,n+b,a,b)
        $bop{T}(::Type{T},::Colon,m::Integer,a::Integer,b::Integer)=$bop(T,m+a,m,a,b)
        $bop(n::Integer,m::Integer,a::Integer,b::Integer)=$bop(Float64,n,m,a,b)
        $bop(n::Integer,a::Integer,b::Integer)=$bop(n,n,a,b)

        $bop{T}(::Type{T},n::Integer,m::Integer,a)=$bop(T,n,m,-a[1],a[end])
        $bop{T}(::Type{T},n::Number,::Colon,a)=$bop(T,n,:,-a[1],a[end])
        $bop{T}(::Type{T},::Colon,m::Integer,a)=$bop(T,:,m,-a[1],a[end])
        $bop{T}(::Type{T},n::Integer,a)=$bop(T,n,-a[1],a[end])
        $bop(n::Integer,m::Integer,a)=$bop(Float64,n,m,-a[1],a[end])
        $bop(n::Integer,a)=$bop(n,-a[1],a[end])

        $bop(B::AbstractBandedMatrix)=$bop(eltype(B),size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))
    end
end

Base.similar(B::AbstractBandedMatrix)=BandedMatrix(eltype(B),size(B,1),size(B,2),bandwidth(B,1),bandwidth(B,2))



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



## Abstract Array Interface

Base.size(A::BandedMatrix,k) = ifelse(k==2,size(A.data,2),A.m)
Base.size(A::BandedMatrix) = A.m,size(A.data,2)

Base.linearindexing{T}(::Type{BandedMatrix{T}})=Base.LinearSlow()


unsafe_getindex(A::BandedMatrix,k::Integer,j::Integer)=A.data[k-j+A.u+1,j]


function getindex(A::BandedMatrix,kr::UnitRange,jr::UnitRange)
    shft=first(kr)-first(jr)
    if A.l-shft ≥ 0 && A.u+shft ≥ 0
        BandedMatrix(A.data[:,jr],length(kr),A.l-shft,A.u+shft)
    else
        #TODO: Make faster
        ret=bzeros(eltype(A),length(kr),length(jr),max(0,A.l-shft),max(0,A.u+shft))
        for (k,j) in eachbandedindex(ret)
            ret[k,j]=A[kr[k],jr[j]]
        end
        ret
    end
end

getindex(A::BandedMatrix,::Colon,jr::UnitRange)=A[1:size(A,1),jr]
getindex(A::BandedMatrix,kr::UnitRange,jr::Colon)=A[kr,1:size(A,2)]
getindex(A::BandedMatrix,kr::Colon,jr::Colon)=copy(A)


# Additional get index overrides
unsafe_getindex(A::BandedMatrix,kr::Range,jr::Integer)=vec(A.data[kr-j+A.u+1,j])

getindex(A::BandedMatrix,kr::Range,j::Integer)=-A.l≤j-kr[1]≤j-kr[end]≤A.u?unsafe_getindex(A,kr,j):[A[k,j] for k=kr]

getindex(A::BandedMatrix,kr::Range,j::Integer)=[A[k,j] for k=kr]
getindex(A::BandedMatrix,kr::Range,jr::Range)=[A[k,j] for k=kr,j=jr]

function getindex(A::BandedMatrix,kr::UnitRange,j::Integer)
    if -A.l≤j-kr[1]≤j-kr[end]≤A.u
        unsafe_getindex(A,kr,j)
    else
        eltype(A)[A[k,j] for k=kr]
    end
end

getindex(A::BandedMatrix,kr::Range,jr::Range)=[A[k,j] for k=kr,j=jr]


# ~~ setindex! ~~

# TODO
# ~ implement indexing with vectors of indices
# ~ implement scalar/vector - band - integer
# ~ implement scalar/vector - band - range

# ~ Utilities ~

# prepare for v0.5 new bound checking facilities
if VERSION < v"0.5"
    macro boundscheck(ex)
        ex
    end
end

# ~~ Type to set\get data along a band
immutable Band
    i::Int
end
band(i::Int) = Band(i)

# ~~ Indexing on the i-th row/column within band range 
immutable BandRange end

# ~~ Out of band error
immutable BandError <: Exception
    A::BandedMatrix
    i::Int
end

function showerror(io::IO, e::BandError)
    A, i, u, l = e.A, e.i, e.A.u, e.A.l
    print(io, "attempt to access $(typeof(A)) with bandwidths " * 
              "($(-l), $u) at band $i")
end

# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline colstart(A::BandedMatrix, i::Integer) = min(max(i-A.u, 1), size(A, 2))
@inline  colstop(A::BandedMatrix, i::Integer) = min(i+A.l, size(A, 1))
@inline rowstart(A::BandedMatrix, i::Integer) = min(max(i-A.l, 1), size(A, 1))
@inline  rowstop(A::BandedMatrix, i::Integer) = min(i+A.u, size(A, 2))

# length of i-the column/row
@inline collength(A::BandedMatrix, i::Integer) = max(colstop(A, i) - colstart(A, i) + 1, 0)
@inline rowlength(A::BandedMatrix, i::Integer) = max(rowstop(A, i) - rowstart(A, i) + 1, 0)

# length of diagonal
@inline diaglength(A::BandedMatrix, b::Band) = diaglength(A, b.i)
@inline function diaglength(A::BandedMatrix, i::Integer)
    min(size(A, 2), size(A, 1)+i) - max(0, i)
end

# return id of lowest-most empty diagonal intersected by row k
function _firstdiagrow(A, k)
    a, b = rowstart(A, k), rowstop(A, k)
    c = a == 1 ? b+1 : a-1
    c-k
end

# return id of lowest-most empty diagonal intersected by column j
function _firstdiagcol(A, j)
    a, b = colstart(A, j), colstop(A, j)
    r = a == 1 ? b+1 : a-1
    j-r
end

# ~ bound checking functions ~

checkbounds(A::BandedMatrix, k::Integer, j::Integer) = 
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::BandedMatrix, kr::Range, j::Integer) = 
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::BandedMatrix, k::Integer, jr::Range) = 
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::BandedMatrix, kr::Range, jr::Range) = 
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

# check indices fall in the band 
checkband(A::BandedMatrix, i::Integer) = 
    (bandinds(A, 1) ≤ i ≤ bandinds(A, 2) || throw(BandError(A, i))) 

checkband(A::BandedMatrix, b::Band) = checkband(A, b.i)

checkband(A::BandedMatrix, k::Integer, j::Integer) = checkband(A, j-k)

checkband(A::BandedMatrix, kr::Range, j::Integer) = 
    (checkband(A, first(kr), j); checkband(A,  last(kr), j))

checkband(A::BandedMatrix, k::Integer, jr::Range) = 
    (checkband(A, k, first(jr)); checkband(A, k,  last(jr)))

checkband(A::BandedMatrix, kr::Range, jr::Range) = 
    (checkband(A, kr, first(jr)); checkband(A, kr,  last(jr)))

# check dimensions of inputs
checkdimensions(sizedest::Tuple{Int, Vararg{Int}}, sizesrc::Tuple{Int, Vararg{Int}}) = 
    (sizedest == sizesrc ||
        throw(DimensionMismatch("tried to assign $(sizesrc) sized " * 
                                "array to $(sizedest) destination")) )

checkdimensions(dest::AbstractVector, src::AbstractVector) = 
    checkdimensions(size(dest), size(src))

checkdimensions(ldest::Int, src::AbstractVector) = 
    checkdimensions((ldest, ), size(src))

checkdimensions(kr::Range, jr::Range, src::AbstractMatrix) = 
    checkdimensions((length(kr), length(jr)), size(src))



# ~ Special setindex methods ~

# slow method to call in a loop as there is a getfield(A, :data) 
@inline unsafe_setindex!{T}(A::BandedMatrix{T}, v, k::Integer, j::Integer) = 
    unsafe_setindex!(A.data, A.u, v, k, j)

# fast method - to be used in loops - )probably can save an addition here)
@inline unsafe_setindex!{T}(data::Matrix{T}, u::Integer, v, k::Integer, j::Integer) = 
    @inbounds data[u + k - j + 1, j] = convert(T, v)::T

# scalar - integer - integer
function setindex!{T}(A::BandedMatrix{T}, v, k::Integer, j::Integer)
    @boundscheck  checkbounds(A, k, j)
    @boundscheck  checkband(A, j-k)
    unsafe_setindex!(A, v, k, j)
end

# scalar - colon - colon
setindex!{T}(A::BandedMatrix{T}, v, ::Colon, ::Colon) =
    @inbounds A.data[:] = convert(T, v)::T 

# scalar - colon
setindex!{T}(A::BandedMatrix{T}, v, ::Colon) =
    @inbounds A.data[:] = convert(T, v)::T 


# ~ indexing along a band

# scalar - band - colon
function setindex!{T}(A::BandedMatrix{T}, v, b::Band, ::Colon)
    @boundscheck checkband(A, b)
    @inbounds A.data[A.u - b.i + 1, :] = convert(T, v)::T
end

# vector - band - colon
function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, b::Band, ::Colon)
    @boundscheck checkband(A, b)
    @boundscheck checkdimensions(diaglength(A, b), V)
    row = A.u - b.i + 1
    data, i = A.data, max(b.i - A.u + 2, 1)
    for v in V
        @inbounds data[row, i] = v
        i += 1
    end
    V
end


# ~ indexing along columns

# scalar - colon - integer -- A[:, 1] = 2 - not allowed
setindex!{T}(A::BandedMatrix{T}, v, kr::Colon, j::Integer) =
    throw(BandError(A, _firstdiagcol(A, j)))

# vector - colon - integer -- A[:, 1] = [1, 2, 3] - not allowed
setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, kr::Colon, j::Integer) = 
    throw(BandError(A, _firstdiagcol(A, j)))

# scalar - BandRange - integer -- A[1, BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, v, ::Type{BandRange}, j::Integer) = 
    (A[colstart(A, j):colstop(A, j), j] = v) # call range method

# vector - BandRange - integer -- A[1, BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, ::Type{BandRange}, j::Integer) = 
    (A[colstart(A, j):colstop(A, j), j] = V) # call range method

# scalar - range - integer -- A[1:2, 1] = 2
function setindex!{T}(A::BandedMatrix{T}, v, kr::Range, j::Integer)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck  checkband(A, kr, j)
    data, u = A.data, A.u
    for k in kr
        unsafe_setindex!(data, u, v, k, j)
    end
    v
end

# vector - range - integer -- A[1:3, 1] = [1, 2, 3]
function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, kr::Range, j::Integer)
    @boundscheck checkbounds(A, kr, j)
    @boundscheck checkdimensions(kr, V)
    @boundscheck  checkband(A, kr, j)
    data, u, i = A.data, A.u, 0
    for v in V
        unsafe_setindex!(data, u, v, kr[i+=1], j)
    end
    V
end


# ~ indexing along a row

# scalar - integer - colon -- A[1, :] = 2 - not allowed
setindex!{T}(A::BandedMatrix{T}, v, k::Integer, jr::Colon) = 
    throw(BandError(A, _firstdiagrow(A, k)))

# vector - integer - colon -- A[1, :] = [1, 2, 3] - not allowed
setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, ::Colon) =
    throw(BandError(A, _firstdiagrow(A, k)))

# scalar - integer - BandRange -- A[1, BandRange] = 2
setindex!{T}(A::BandedMatrix{T}, v, k::Integer, ::Type{BandRange}) = 
    (A[k, rowstart(A, k):rowstop(A, k)] = v) # call range method

# vector - integer - BandRange -- A[1, BandRange] = [1, 2, 3]
setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, ::Type{BandRange}) =
    (A[k, rowstart(A, k):rowstop(A, k)] = V) # call range method

# scalar - integer - range -- A[1, 2:3] = 3
function setindex!{T}(A::BandedMatrix{T}, v, k::Integer, jr::Range)
    @boundscheck checkbounds(A, k, jr)
    @boundscheck checkband(A, k, jr)
    data, u = A.data, A.u
    for j in jr
        unsafe_setindex!(data, u, v, k, j)
    end
    v
end

# vector - integer - range -- A[1, 2:3] = [3, 4]
function setindex!{T}(A::BandedMatrix{T}, V::AbstractVector, k::Integer, jr::Range)
    @boundscheck checkbounds(A, k, jr)
    @boundscheck checkband(A, k, jr)
    @boundscheck checkdimensions(jr, V)
    data, u, i = A.data, A.u, 0
    for v in V
        unsafe_setindex!(data, u, v, k, jr[i+=1])
    end
    V
end

# ~ indexing over a rectangular block

# scalar - range - range
function setindex!{T}(A::BandedMatrix{T}, v, kr::Range, jr::Range)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkband(A, kr, jr)
    data, u = A.data, A.u
    for j in jr, k in kr
        unsafe_setindex!(data, u, v, k, j)
    end
    v
end    

# matrix - range - range
function setindex!{T}(A::BandedMatrix{T}, V::AbstractMatrix, kr::Range, jr::Range)
    @boundscheck checkbounds(A, kr, jr)
    @boundscheck checkband(A, kr, jr)
    @boundscheck checkdimensions(kr, jr, V)
    data, u = A.data, A.u
    jj = 1
    for j in jr
        kk = 1
        for k in kr
            # we index V manually in column-major order
            @inbounds unsafe_setindex!(data, u, V[kk, jj], k, j)
            kk += 1
        end
        jj += 1
    end
    V
end   

# ~~ end setindex! ~~



function Base.full(A::BandedMatrix)
    ret=zeros(eltype(A),size(A,1),size(A,2))
    for (k,j) in eachbandedindex(A)
        @inbounds ret[k,j]=A[k,j]
    end
    ret
end


## Band range

bandwidth(A::BandedMatrix,k)=k==1?A.l:A.u


function Base.sparse(B::BandedMatrix)
    i=Array(Int,length(B.data));j=Array(Int,length(B.data))
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

Base.norm(B::BandedMatrix,opts...)=norm(full(B),opts...)


# We turn off bound checking to allow nicer syntax without branching
#setindex!(A::BandedMatrix,v,k::Integer,j::Integer)=((A.l≤j-k≤A.u)&&k≤A.n)?ussetindex!(A,v,k,j):throw(BoundsError())
#setindex!(A::BandedMatrix,v,kr::Range,j::Integer)=(A.l≤j-kr[end]≤j-kr[1]≤A.u&&kr[end]≤A.n)?ussetindex!(A,v,kr,j):throw(BoundsError())


## ALgebra and other functions

function Base.maximum(B::BandedMatrix)
    m=zero(eltype(B))
    for (k,j) in eachbandedindex(B)
        m=max(B[k,j],m)
    end
    m
end


for OP in (:*,:.*,:+,:.+,:-,:.-)
    @eval begin
        $OP(B::BandedMatrix{Bool},x::Bool) = BandedMatrix($OP(B.data,x),B.m,B.l,B.u)
        $OP(x::Bool,B::BandedMatrix{Bool}) = BandedMatrix($OP(x,B.data),B.m,B.l,B.u)
        $OP(B::BandedMatrix,x::Number) = BandedMatrix($OP(B.data,x),B.m,B.l,B.u)
        $OP(x::Number,B::BandedMatrix) = BandedMatrix($OP(x,B.data),B.m,B.l,B.u)
    end
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

function -{T,V}(A::BandedMatrix{T},B::BandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(-1.,B,ret)

    ret
end


function *(A::BandedMatrix,B::BandedMatrix)
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)

    ret=BandedMatrix(promote_type(eltype(A),eltype(B)),n,m,A.l+B.l,A.u+B.u)
    for (k,j) in eachbandedindex(ret)
        νmin=max(1,k-bandwidth(A,1),j-bandwidth(B,2))
        νmax=min(size(A,2),k+bandwidth(A,2),j+bandwidth(B,1))

        ret[k,j]=A[k,νmin]*B[νmin,j]
        for ν=νmin+1:νmax
            ret[k,j]+=A[k,ν]*B[ν,j]
        end
    end

    ret
end



function *{T<:Number,V<:Number}(A::BandedMatrix{T},B::BandedMatrix{V})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)

    A_mul_B!(bzeros(promote_type(T,V),n,m,A.l+B.l,A.u+B.u),A,B)
end

function *{T<:Number,V<:Number}(A::BandedMatrix{T},B::Matrix{V})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)

    A_mul_B!(Array(promote_type(T,V),n,m),A,B)
end



*{T<:BlasFloat}(A::BandedMatrix{T},b::Vector{T}) =
    BLAS.gbmv('N',A.m,A.l,A.u,one(T),A.data,b)

function *{T}(A::BandedMatrix{T},b::Vector{T})
    ret = zeros(T,size(A,1))
    for (k,j) in eachbandedindex(A)
        @inbounds ret[k]+=A[k,j]*b[j]
    end
    ret
end


function *(A::BandedMatrix,b::Vector)
    T=promote_type(eltype(A),eltype(b))
    convert(BandedMatrix{T},A)*convert(Vector{T},b)
end

function Base.transpose(B::BandedMatrix)
    Bt=bzeros(eltype(B),size(B,2),size(B,1),B.u,B.l)
    for (k,j) in eachbandedindex(B)
       Bt[j,k]=B[k,j]
    end
    Bt
end

function Base.ctranspose(B::BandedMatrix)
    Bt=bzeros(eltype(B),size(B,2),size(B,1),B.u,B.l)
    for (k,j) in eachbandedindex(B)
       Bt[j,k]=conj(B[k,j])
    end
    Bt
end



function Base.diag{T}(A::BandedMatrix{T})
    n=size(A,1)
    @assert n==size(A,2)

    vec(A.data[A.u+1,1:n])
end



# basic interface
(\){T<:BlasFloat}(A::Union{BandedLU{T}, BandedMatrix{T}}, B::StridedVecOrMat{T}) =
    A_ldiv_B!(A, copy(B)) # makes a copy


## Matrix.*Matrix

function .*(A::BandedMatrix,B::BandedMatrix)
    @assert size(A,1)==size(B,1)&&size(A,2)==size(B,2)

    l=min(A.l,B.l);u=min(A.u,B.u)
    T=promote_type(eltype(A),eltype(B))
    ret=BandedMatrix(T,size(A,1),size(A,2),l,u)

    for (k,j) in eachbandedindex(ret)
        @inbounds ret[k,j]=A[k,j]*B[k,j]
    end
    ret
end


#implements fliplr(flipud(A))
function fliplrud(A::BandedMatrix)
    n,m=size(A)
    l=A.u+n-m
    u=A.l+m-n
    ret=BandedMatrix(eltype(A),n,m,l,u)
    for (k,j) in eachbandedindex(ret)
        @inbounds ret[k,j]=A[n-k+1,m-j+1]
    end
    ret
end


for OP in (:(Base.real),:(Base.imag))
    @eval $OP(A::BandedMatrix) =
        BandedMatrix($OP(A.data),A.m,A.l,A.u)
end



## Show

type PrintShow
    str
end
Base.show(io::IO,N::PrintShow) = print(io,N.str)

#TODO: implement showarray in 0.5

if VERSION < v"0.5.0-dev"
    function Base.showarray(io::IO,B::AbstractBandedMatrix;
                       header::Bool=true, limit::Bool=Base._limit_output,
                       sz = (s = Base.tty_size(); (s[1]-4, s[2])), repr=false)
        header && print(io,summary(B))

        if !isempty(B) && size(B,1) ≤ 1000 && size(B,2) ≤ 1000
            header && println(io,":")
            M=Array(Any,size(B)...)
            fill!(M,PrintShow(""))
            for (k,j) in eachbandedindex(B)
                M[k,j]=B[k,j]
            end

            Base.showarray(io,M;header=false)
        end
    end
end



## SubArray routines

bandshift(S) = first(parentindexes(S)[1])-first(parentindexes(S)[2])

bandwidth{T,BM<:BandedMatrix}(S::SubArray{T,2,BM},k) = bandwidth(parent(S),k) +
        (k==1?-1:1)*bandshift(S)


unsafe_getindex{T,BM<:BandedMatrix}(S::SubArray{T,2,BM},k,j) =
    unsafe_getindex(parent(S),parentindexes(S)[1][k],parentindexes(S)[2][j])



end #module
