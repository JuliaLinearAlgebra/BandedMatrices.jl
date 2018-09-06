####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedColumnMajor, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

abstract type AbstractBandedLayout <: MemoryLayout end
struct BandedColumnMajor <: AbstractBandedLayout end
struct BandedRowMajor <: AbstractBandedLayout end

transposelayout(::BandedColumnMajor) = BandedRowMajor()
transposelayout(::BandedRowMajor) = BandedColumnMajor()
conjlayout(::Type{<:Complex}, M::AbstractBandedLayout) = ConjLayout(M)

# Here we override broadcasting for banded matrices.
# The design is to to exploit the broadcast machinery so that
# banded matrices that conform to the banded matrix interface but are not
# <: AbstractBandedMatrix can get access to fast copyto!, lmul!, rmul!, axpy!, etc.
# using broadcast variants (B .= A, B .= 2.0 .* A, etc.)



struct BandedStyle <: AbstractArrayStyle{2} end
BandedStyle(::Val{2}) = BandedStyle()
BroadcastStyle(::Type{<:AbstractBandedMatrix}) = BandedStyle()
BroadcastStyle(::DefaultArrayStyle{2}, ::BandedStyle) = DefaultArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::DefaultArrayStyle{2}) = DefaultArrayStyle{2}()



####
# Default to standard Array broadcast
#
# This is because, for example, exp.(B) is not banded
####


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle}) =
   copyto!(dest, Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes))

similar(bc::Broadcasted{BandedStyle}, ::Type{T}) where T =
    similar(Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes), T)

##
# copyto!
##

copyto!(dest::AbstractMatrix, src::AbstractBandedMatrix) =  banded_copyto!(dest, src)

function banded_copyto!(dest::AbstractMatrix{T}, src::AbstractMatrix) where T
    m,n = size(dest)
    (m,n) == size(src) || throw(DimensionMismatch())

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)
    (d_l ≥ min(s_l,m-1) && d_u ≥ min(s_u,n-1)) || throw(BandError(dest))
    for j=1:n
        for k = max(1,j-d_u):min(j-s_u-1,m)
            inbands_setindex!(dest, zero(T), k, j)
        end
        for k = max(1,j-s_u):min(j+s_l,m)
            inbands_setindex!(dest, inbands_getindex(src, k, j), k, j)
        end
        for k = max(1,j+s_l+1):min(j+d_l,m)
            inbands_setindex!(dest, zero(T), k, j)
        end
    end
    dest
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(identity)})
    (A,) = bc.args
    dest ≡ A && return dest
    banded_copyto!(dest, A)
end


##
# lmul!/rmul!
##

function _banded_lmul!(α, A::AbstractMatrix, _)
    for j=1:size(A,2), k=colrange(A,j)
        inbands_setindex!(A, α*inbands_getindex(A,k,j), k,j)
    end
    A
end

function _banded_rmul!(A::AbstractMatrix, a, _)
    for j=1:size(Α,2), k=colrange(A,j)
        inbands_setindex!(A, inbands_getindex(A,k,j)*α, k,j)
    end
    A
end


function _banded_lmul!(α::Number, A::AbstractMatrix, ::BandedColumnMajor)
    lmul!(α, bandeddata(A))
    A
end

function _banded_rmul!(A::AbstractMatrix, α::Number, ::BandedColumnMajor)
    rmul!(bandeddata(A), α)
    A
end

banded_lmul!(α, A::AbstractMatrix) = _banded_lmul!(α, A, MemoryLayout(A))

banded_rmul!(A::AbstractMatrix, α) = _banded_rmul!(A, α, MemoryLayout(A))

lmul!(α::Number, A::AbstractBandedMatrix) = banded_lmul!(α, A)
rmul!(A::AbstractBandedMatrix, α::Number) = banded_rmul!(A, α)


function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(-), <:Tuple{<:AbstractMatrix}})
    A, = bc.args
    dest ≡ A || copyto!(dest, A)
    banded_lmul!(-1, dest)
end

similar(bc::Broadcasted{BandedStyle, <:Any, typeof(-), <:Tuple{<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(*), <:Tuple{<:Number,<:AbstractMatrix}})
    α,A = bc.args
    dest ≡ A || copyto!(dest, A)
    banded_lmul!(α, dest)
end

similar(bc::Broadcasted{BandedStyle, <:Any, typeof(*), <:Tuple{<:Number,<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[2], T)


function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(*), <:Tuple{<:AbstractMatrix,<:Number}})
    A,α = bc.args
    dest ≡ A || copyto!(dest, A)
    banded_rmul!(dest, α)
end

similar(bc::Broadcasted{BandedStyle, <:Any, typeof(*), <:Tuple{<:AbstractMatrix,<:Number}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(/), <:Tuple{<:AbstractMatrix,<:Number}})
    A,α = bc.args
    dest ≡ A || copyto!(dest, A)
    banded_rmul!(dest, inv(α))
end

similar(bc::Broadcasted{BandedStyle, <:Any, typeof(/), <:Tuple{<:AbstractMatrix,<:Number}}, ::Type{T}) where T =
    similar(bc.args[1], T)

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(\), <:Tuple{<:Number,<:AbstractMatrix}})
    α,A = bc.args
    dest ≡ A || copyto!(dest, A)
    banded_lmul!(inv(α), dest)
end

similar(bc::Broadcasted{BandedStyle, <:Any, typeof(\), <:Tuple{<:Number,<:AbstractMatrix}}, ::Type{T}) where T =
    similar(bc.args[2], T)

##
# axpy!
##

# these are the routines of the banded interface of other AbstractMatrices
banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix) = _banded_axpy!(a, X, Y, MemoryLayout(X), MemoryLayout(Y))
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, ::BandedColumnMajor, ::BandedColumnMajor) =
    banded_generic_axpy!(a, X, Y)
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, notbandedX, notbandedY) =
    banded_dense_axpy!(a, X, Y)

# additions and subtractions
@propagate_inbounds function banded_generic_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    n,m = size(X)
    if (n,m) ≠ size(Y)
        throw(BoundsError())
    end
    Xl, Xu = bandwidths(X)
    Yl, Yu = bandwidths(Y)

    @boundscheck if Xl > Yl
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j+Yl+1):min(j+Xl,n)
            if inbands_getindex(X, k, j) ≠ 0
                throw(BandError(X, (k,j)))
            end
        end
    end
    @boundscheck if Xu > Yu
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j-Xu):min(j-Yu-1,n)
            if inbands_getindex(X, k, j) ≠ 0
                throw(BandError(X, (k,j)))
            end
        end
    end

    l = min(Xl,Yl)
    u = min(Xu,Yu)

    @inbounds for j=1:m,k=max(1,j-u):min(n,j+l)
        inbands_setindex!(Y, a*inbands_getindex(X,k,j) + inbands_getindex(Y,k,j) ,k, j)
    end
    Y
end

function banded_dense_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    if size(X) != size(Y)
        throw(DimensionMismatch("+"))
    end
    @inbounds for j=1:size(X,2),k=colrange(X,j)
        Y[k,j] += a*inbands_getindex(X,k,j)
    end
    Y
end


function copyto!(dest::AbstractArray{T}, bc::Broadcasted{BandedStyle, <:Any, typeof(+),
                                                            <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}) where T
    A,B = bc.args
    if dest ≡ B
        banded_axpy!(one(T), A, dest)
    elseif dest ≡ A
        banded_axpy!(one(T), B, dest)
    else
        banded_copyto!(dest, B)
        banded_axpy!(one(T), A, dest)
    end
end

function similar(bc::Broadcasted{BandedStyle, <:Any, typeof(+), <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}, ::Type{T}) where T
    A,B = bc.args
    n,m = size(A)
    (n,m) == size(B) || throw(DimensionMismatch())
    Al,Au = bandwidths(A)
    Bl,Bu = bandwidths(B)
    similar(A, T, n, m, max(Al,Bl), max(Au,Bu))
end


function copyto!(dest::AbstractArray{T}, bc::Broadcasted{BandedStyle, <:Any, typeof(+),
                                                        <:Tuple{<:Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
                                                        <:AbstractMatrix}}) where T
    αA,B = bc.args
    α,A = αA.args
    dest ≡ B || banded_copyto!(dest, B)
    banded_axpy!(α, A, dest)
end

function similar(bc::Broadcasted{BandedStyle, <:Any, typeof(+),
                        <:Tuple{<:Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
                        <:AbstractMatrix}}, ::Type{T}) where T
    αA,B = bc.args
    α,A = αA.args
    n,m = size(A)
    (n,m) == size(B) || throw(DimensionMismatch())
    Al,Au = bandwidths(A)
    Bl,Bu = bandwidths(B)
    similar(A, T, n, m, max(Al,Bl), max(Au,Bu))
end
