__precompile__()

module BandedMatrices
    using Base


import Base: getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,transpose

export BandedMatrix, bandrange, bzeros,beye,brand,bones



##
# Represent a banded matrix
# [ a_11 a_12
#   a_21 a_22 a_23
#   a_31 a_32 a_33 a_34
#        a_42 a_43 a_44  ]
# ordering the data like (rows first)
#       [ *     *       a_31    a_42
#         *      a_21   a_32    a_43
#         a_11   a_22   a_33    A_44
#         a_12   a_23   a_34    *       ]
#  (columns first)
#       [ *      a_12   a_23    a_34
#         a_11   a_22   a_33    a_43
#         a_21   a_32   a_43    *
#         a_32   a_42   *       *       ]
###


type BandedMatrix{colrow,T} <: AbstractSparseMatrix{T,Int}
    data::Matrix{T}  # l+u+1 x n (# of rows)
    m::Int #Number of rows ('C') or columns ('R')
    l::Int # lower bandwidth ≥0
    u::Int # upper bandwidth ≥0
    function BandedMatrix(data::Matrix{T},m,l,u)
        @assert size(data,1)==l+u+1
        new(data,m,l,u)
    end
end


include("blas.jl")


BandedMatrix{cr}(::Type{Val{cr}},data::Matrix,m::Integer,a::Integer,b::Integer) = BandedMatrix{cr,eltype(data)}(data,m,a,b)
BandedMatrix(data::Matrix,m::Integer,a::Integer,b::Integer) = BandedMatrix{'R',eltype(data)}(data,m,a,b)

BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) = BandedMatrix{'R',T}(Array(T,b+a+1,n),m,a,b)
BandedMatrix{T}(::Type{T},n::Integer,a::Integer,b::Integer) = BandedMatrix(T,n,n,a,b)
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) = BandedMatrix(T,n,n+b,a,b)


BandedMatrix(data::Matrix,m::Integer,a) = BandedMatrix(data,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,m::Integer,a) = BandedMatrix(T,n,m,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,::Colon,a) = BandedMatrix(T,n,:,-a[1],a[end])
BandedMatrix{T}(::Type{T},n::Integer,a) = BandedMatrix(T,n,-a[1],a[end])


Base.convert{cr,V}(::Type{BandedMatrix{cr,V}},M::BandedMatrix{cr}) = BandedMatrix{cr,V}(convert(Matrix{V},M.data),M.m,M.l,M.u)
# function Base.convert{BM<:BandedMatrix}(::Type{BM},M::Matrix)
#     ret=BandedMatrix(eltype(BM)==Any?eltype(M):promote_type(eltype(BM),eltype(M)),size(M,1),size(M,2),size(M,1)-1,size(M,2)-1)
#     for k=1:size(M,1),j=1:size(M,2)
#         ret[k,j]=M[k,j]
#     end
#     ret
# end


Base.promote_rule{cr,T,V}(::Type{BandedMatrix{cr,T}},::Type{BandedMatrix{cr,V}})=BandedMatrix{cr,promote_type(T,V)}



for (op,bop) in ((:(Base.rand),:brand),(:(Base.zeros),:bzeros),(:(Base.ones),:bones))
    @eval begin
        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,m::Integer,a::Integer,b::Integer)=BandedMatrix(Val{cr},$op(T,b+a+1,n),m,a,b)
        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,a::Integer,b::Integer)=$bop(Val{cr},T,n,n,a,b)
        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,::Colon,a::Integer,b::Integer)=$bop(Val{cr},T,n,n+b,a,b)
        $bop{cr,T}(::Type{Val{cr}},::Type{T},::Colon,m::Integer,a::Integer,b::Integer)=$bop(Val{cr},T,m+a,m,a,b)
        $bop{cr}(::Type{Val{cr}},n::Integer,m::Integer,a::Integer,b::Integer)=$bop(Val{cr},Float64,n,m,a,b)
        $bop{cr}(::Type{Val{cr}},n::Integer,a::Integer,b::Integer)=$bop(Val{cr},n,n,a,b)

        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,m::Integer,a)=$bop(Val{cr},T,n,m,-a[1],a[end])
        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Number,::Colon,a)=$bop(Val{cr},T,n,:,-a[1],a[end])
        $bop{cr,T}(::Type{Val{cr}},::Type{T},::Colon,m::Integer,a)=$bop(Val{cr},T,:,m,-a[1],a[end])
        $bop{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,a)=$bop(Val{cr},T,n,-a[1],a[end])
        $bop{cr}(::Type{Val{cr}},n::Integer,m::Integer,a)=$bop(Val{cr},Float64,n,m,-a[1],a[end])
        $bop{cr}(::Type{Val{cr}},n::Integer,a)=$bop(Val{cr},n,-a[1],a[end])

        $bop(x...)=$bop(Val{'R'},x...)
    end
end



function beye{cr,T}(::Type{Val{cr}},::Type{T},n::Integer,a...)
    ret=bzeros(Val{cr},T,n,a...)
    for k=1:n
         ret[k,k]=one(T)
    end
    ret
end
beye{cr,T}(::Type{Val{cr}},::Type{T},n::Integer) = beye(Val{cr},T,n,0,0)
beye{cr}(::Type{Val{cr}},n::Integer) = beye(Val{cr},n,0,0)
beye{cr}(::Type{Val{cr}},n::Integer,a...) = beye(Val{cr},Float64,n,a...)

beye(x...) = beye(Val{'R'},x...)


Base.size(A::BandedMatrix{'C'},k) = ifelse(k==2,size(A.data,2),A.m)
Base.size(A::BandedMatrix{'C'}) = A.m,size(A.data,2)
Base.size(A::BandedMatrix{'R'},k) = ifelse(k==1,size(A.data,2),A.m)
Base.size(A::BandedMatrix{'R'}) = size(A.data,2),A.m


bandinds(A::BandedMatrix) = -A.l,A.u
bandrange(A::BandedMatrix) = -A.l:A.u




## Gives an iterator over the banded entries of a BandedMatrix

immutable BandedIterator{colrow}
    n::Int
    m::Int
    l::Int
    u::Int
end


Base.start(B::BandedIterator)=(1,1)
Base.next(B::BandedIterator{'R'},state)=
    state,ifelse(state[2]==min(state[1]+B.u,B.m),
                (state[1]+1,max(state[1]+1-B.l,1)),
                (state[1],  state[2]+1)
                 )

Base.next(B::BandedIterator{'C'},state)=
    state,ifelse(state[1]==min(state[2]+B.l,B.m),
                (max(state[2]+1-B.u,1),state[2]+1),
                (state[1]+1,  state[2])
                 )

Base.done(B::BandedIterator{'R'},state)=state[1]>B.n || state[1]>B.m+B.l
Base.done(B::BandedIterator{'C'},state)=state[2]>B.m || state[2]>B.n+B.u

Base.eltype{cr}(::Type{BandedIterator{cr}})=Tuple{Int,Int}
# the length is independent of where its 'C' or 'R'
function Base.length(B::BandedIterator)
    if B.m > B.n
        p=max(0,B.u+B.n-B.m)
        B.n*(B.u+1)+
            div(B.l*(2*B.n-B.l-1),2)-div(p*(1+p),2)
    else
        p=max(0,B.l+B.m-B.n)
        B.m*(B.l+1)+
            div(B.u*(2*B.m-B.u-1),2)-div(p*(1+p),2)
    end
end

# returns an iterator of each index in the banded part of a matrix
eachbandedindex{cr}(B::BandedMatrix{cr})=BandedIterator{cr}(size(B,1),size(B,2),B.l,B.u)



unsafe_getindex(A::BandedMatrix{'R'},k::Integer,j::Integer)=A.data[j-k+A.l+1,k]
unsafe_getindex(A::BandedMatrix{'R'},k::Integer,jr::Range)=vec(A.data[jr-k+A.l+1,k])

unsafe_getindex(A::BandedMatrix{'C'},k::Integer,j::Integer)=A.data[k-j+A.u+1,j]
unsafe_getindex(A::BandedMatrix{'C'},kr::Range,jr::Integer)=vec(A.data[kr-j+A.u+1,j])


function getindex(A::BandedMatrix,k::Integer,j::Integer)
    if k>size(A,1) || j>size(A,2)
        throw(BoundsError())
    elseif (-A.l≤j-k≤A.u)
        unsafe_getindex(A,k,j)
    else
        zero(eltype(A))
    end
end

getindex(A::BandedMatrix{'R'},k::Integer,jr::Range)=-A.l≤jr[1]-k≤jr[end]-k≤A.u?unsafe_getindex(A,k,jr):[A[k,j] for j=jr].'
getindex(A::BandedMatrix{'C'},kr::Range,j::Integer)=-A.l≤j-kr[1]≤j-kr[end]≤A.u?unsafe_getindex(A,kr,j):[A[k,j] for k=kr]

getindex(A::BandedMatrix,kr::Range,j::Integer)=[A[k,j] for k=kr]
getindex(A::BandedMatrix,kr::Range,jr::Range)=[A[k,j] for k=kr,j=jr]
Base.full(A::BandedMatrix)=A[1:size(A,1),1:size(A,2)]


function Base.sparse(B::BandedMatrix{'R'})
    i=Array(Int,length(B.data));j=Array(Int,length(B.data))
    n,m=size(B.data)
    Bm=size(B,2)
    vb=copy(vec(B.data))
    for k=1:n,ℓ=1:m
        i[k+n*(ℓ-1)]=ℓ
        jj=k+ℓ-B.l-1
        if jj <1 || jj > Bm
            vb[k+n*(ℓ-1)] = 0
        end
        j[k+n*(ℓ-1)]=min(max(jj,1),Bm)
    end
    sparse(i,j,vb)
end




# pass standard routines to full matrix

Base.norm(B::BandedMatrix,opts...)=norm(full(B),opts...)


# We turn off bound checking to allow nicer syntax without branching
#setindex!(A::BandedMatrix,v,k::Integer,j::Integer)=((A.l≤j-k≤A.u)&&k≤A.n)?ussetindex!(A,v,k,j):throw(BoundsError())
#setindex!(A::BandedMatrix,v,kr::Range,j::Integer)=(A.l≤j-kr[end]≤j-kr[1]≤A.u&&kr[end]≤A.n)?ussetindex!(A,v,kr,j):throw(BoundsError())


unsafe_setindex!(A::BandedMatrix{'R'},v,k::Integer,j::Integer)=(@inbounds A.data[j-k+A.l+1,k]=v)
unsafe_setindex!(A::BandedMatrix{'C'},v,k::Integer,j::Integer)=(@inbounds A.data[k-j+A.u+1,j]=v)


"unsafe_pluseq!(A,v,k,j) is an unsafe versoin of A[k,j] += v"
unsafe_pluseq!(A::BandedMatrix{'R'},v,k::Integer,j::Integer)=(@inbounds A.data[j-k+A.l+1,k]+=v)
unsafe_pluseq!(A::BandedMatrix{'C'},v,k::Integer,j::Integer)=(@inbounds A.data[k-j+A.u+1,j]+=v)


setindex!(A::BandedMatrix{'R'},v,k::Integer,j::Integer)=(A.data[j-k+A.l+1,k]=v)
setindex!(A::BandedMatrix{'C'},v,k::Integer,j::Integer)=(A.data[k-j+A.u+1,j]=v)

function setindex!(A::BandedMatrix,v,kr::Range,jr::Range)
    for j in jr
        A[kr,j]=slice(v,:,j)
    end
end
function setindex!(A::BandedMatrix,v,k::Integer,jr::Range)
    for j in jr
        A[k,j]=v[j]
    end
end



## ALgebra and other funcrtions

function Base.maximum(B::BandedMatrix)
    m=zero(eltype(B))
    for (k,j) in eachbandedindex(B)
        m=max(B[k,j],m)
    end
    m
end


for OP in (:*,:.*,:+,:.+,:-,:.-)
    @eval begin
        $OP{cr}(B::BandedMatrix{cr,Bool},x::Bool) = BandedMatrix(Val{cr},$OP(B.data,x),B.m,B.l,B.u)
        $OP{cr}(x::Bool,B::BandedMatrix{cr,Bool}) = BandedMatrix(Val{cr},$OP(x,B.data),B.m,B.l,B.u)
        $OP{cr}(B::BandedMatrix,x::Number) = BandedMatrix(Val{cr},$OP(B.data,x),B.m,B.l,B.u)
        $OP{cr}(x::Number,B::BandedMatrix{cr}) = BandedMatrix(Val{cr},$OP(x,B.data),B.m,B.l,B.u)
    end
end

function +{cr,T,V}(A::BandedMatrix{cr,T},B::BandedMatrix{cr,V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(Val{cr},promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(1.,B,ret)

    ret
end

function -{cr,T,V}(A::BandedMatrix{cr,T},B::BandedMatrix{cr,V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(Val{cr},promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(-1.,B,ret)

    ret
end


function *{cr}(A::BandedMatrix{cr},B::BandedMatrix{cr})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)
    T=promote_type(eltype(A),eltype(B))
    A_mul_B!(bzeros(Val{cr},T,n,m,A.l+B.l,A.u+B.u),A,B)
end

function *{cr}(A::BandedMatrix{cr},B::Matrix{cr})
    if size(A,2)!=size(B,1)
        throw(DimensionMismatch("*"))
    end
    n,m=size(A,1),size(B,2)
    T=promote_type(eltype(A),eltype(B))
    A_mul_B!(Array(T,n,m),A,B)
end


*{T}(A::BandedMatrix{'R',T},b::Vector{T})=BLAS.gbmv('T',A.m,A.u,A.l,one(T),A.data,b)
*{T}(A::BandedMatrix{'C',T},b::Vector{T})=BLAS.gbmv('N',A.n,A.l,A.u,one(T),A.data,b)

function *{cr}(A::BandedMatrix{cr},b::Vector)
    T=promote_type(eltype(A),eltype(b))
    convert(BandedMatrix{cr,T},A)*convert(Vector{T},b)
end

function Base.transpose{cr}(B::BandedMatrix{cr})
    Bt=bzeros(Val{cr},eltype(B),size(B,2),size(B,1),B.u,B.l)
    for (k,j) in eachbandedindex(B)
       Bt[j,k]=B[k,j]
    end
    Bt
end

function Base.ctranspose{cr}(B::BandedMatrix{cr})
    Bt=bzeros(Val{cr},eltype(B),size(B,2),size(B,1),B.u,B.l)
    for (k,j) in eachbandedindex(B)
       Bt[j,k]=conj(B[k,j])
    end
    Bt
end


function Base.diag{T}(A::BandedMatrix{'R',T})
    n=size(A,1)
    @assert n==size(A,2)

    vec(A.data[A.l+1,1:n])
end

function Base.diag{T}(A::BandedMatrix{'C',T})
    n=size(A,1)
    @assert n==size(A,2)

    vec(A.data[A.u+1,1:n])
end




## Matrix.*Matrix

function .*{cr}(A::BandedMatrix{cr},B::BandedMatrix{cr})
    @assert size(A,1)==size(B,1)&&size(A,2)==size(B,2)

    l=min(A.l,B.l);u=min(A.u,B.u)
    T=promote_type(eltype(A),eltype(B))
    ret=BandedMatrix(Val{cr},T,size(A,1),size(A,2),l,u)

    for (k,j) in eachbandedindex(ret)
        @inbounds ret[k,j]=A[k,j]*B[k,j]
    end
    ret
end


#implements fliplr(flipud(A))
function fliplrud{cr}(A::BandedMatrix{cr})
    n,m=size(A)
    l=A.u+n-m
    u=A.l+m-n
    ret=BandedMatrix(Val{cr},eltype(A),n,m,l,u)
    for (k,j) in eachbandedindex(ret)
        @inbounds ret[k,j]=A[n-k+1,m-j+1]
    end
    ret
end



## Show

type PrintShow
    str
end
Base.show(io::IO,N::PrintShow)=print(io,N.str)

function Base.showarray(io::IO,B::BandedMatrix;
                   header::Bool=true, limit::Bool=Base._limit_output,
                   sz = (s = Base.tty_size(); (s[1]-4, s[2])), repr=false)
    header && print(io,summary(B))

    if !isempty(B)
        header && println(io,":")
        M=Array(Any,size(B)...)
        fill!(M,PrintShow(""))
        for (k,j) in eachbandedindex(B)
            M[k,j]=B[k,j]
        end

        Base.showarray(io,M;header=false)
    end
end


end #module
