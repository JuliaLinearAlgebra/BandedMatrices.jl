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



size(bc::Broadcasted{BandedStyle}) = length.(axes(bc))

####
# Default to standard Array broadcast
#
# This is because, for example, exp.(B) is not banded
####


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle}) =
   copyto!(dest, Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes))
##
# copyto!
##

copyto!(dest::AbstractMatrix, src::AbstractBandedMatrix) =  (dest .= src)


function checkbroadcastband(dest, src::AbstractMatrix, z)
    iszero(z) || (size(src,1) ≤ bandwidth(src,1)+1 && size(src,2) ≤ bandwidth(src,2)+1) ||
        (size(dest,1) ≤ bandwidth(dest,1)+1 && size(dest,2) ≤ bandwidth(dest,2)+1) || throw(BandError(dest,size(dest,2)-1))
    size(dest) == size(src) || throw(DimensionMismatch())
end


function checkbroadcastband(dest, (A,B)::Tuple{AbstractMatrix,AbstractMatrix}, z)
    iszero(z) || (size(dest,1) ≤ bandwidth(dest,1)+1 && size(dest,2) ≤ bandwidth(dest,2)+1) || throw(BandError(dest,size(dest,2)-1))
    size(dest) == size(A) == size(B) || throw(DimensionMismatch())
end

function _banded_broadcast!(dest::AbstractMatrix, f, src::AbstractMatrix{T}, _1, _2) where T
    z = f(zero(T))
    checkbroadcastband(dest, src, z)
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)
    (d_l ≥ min(s_l,m-1) && d_u ≥ min(s_u,n-1)) || throw(BandError(dest))

    for j=1:n
        for k = max(1,j-d_u):min(j-s_u-1,m)
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-s_u):min(j+s_l,m)
            inbands_setindex!(dest, f(inbands_getindex(src, k, j)), k, j)
        end
        for k = max(1,j+s_l+1):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (src,x)::Tuple{AbstractMatrix{T},Number}, _1, _2) where T
    z = f(zero(T), x)
    checkbroadcastband(dest, src, z)
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)
    (d_l ≥ min(s_l,m-1) && d_u ≥ min(s_u,n-1)) || throw(BandError(dest))

    for j=1:n
        for k = max(1,j-d_u):min(j-s_u-1,m)
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-s_u):min(j+s_l,m)
            inbands_setindex!(dest, f(inbands_getindex(src, k, j), x), k, j)
        end
        for k = max(1,j+s_l+1):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (x,src)::Tuple{Number,AbstractMatrix{T}}, _1, _2) where T
    z = f(x, zero(T))
    checkbroadcastband(dest, src, z)
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)
    (d_l ≥ min(s_l,m-1) && d_u ≥ min(s_u,n-1)) || throw(BandError(dest))

    for j=1:n
        for k = max(1,j-d_u):min(j-s_u-1,m)
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-s_u):min(j+s_l,m)
            inbands_setindex!(dest, f(x, inbands_getindex(src, k, j)), k, j)
        end
        for k = max(1,j+s_l+1):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _1, _2) where {T,V}
    z = f(zero(T), zero(V))
    checkbroadcastband(dest, (A,B), z)
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    A_l, A_u = bandwidths(A)
    B_l, B_u = bandwidths(B)
    l, u = max(A_l,B_l), max(A_u,B_u)
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    for j=1:n
        for k = max(1,j-d_u):min(j-u-1,m)
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-A_u):min(j-B_u-1,m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
        end
        for k = max(1,j-B_u):min(j-A_u-1,m)
            inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j-min(A_u,B_u)):min(j+min(A_l,B_l),m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j+B_l+1):min(j+A_l,m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
        end
        for k = max(1,j+A_l+1):min(j+B_l,m)
            inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j+l+1):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
    dest
end


function _banded_broadcast!(dest::AbstractMatrix, f, src::AbstractMatrix{T}, ::BandedColumnMajor, ::BandedColumnMajor) where T
    z = f(zero(T))
    checkbroadcastband(dest, src, z)

    l,u = bandwidths(src)
    λ,μ = bandwidths(dest)
    m,n = size(src)
    data_d,data_s = bandeddata(dest), bandeddata(src)
    if (l,u) == (λ,μ)
        data_d .= f.(data_s)
    elseif μ > u && λ > l
        fill!(view(data_d,1:(μ-u),:), z)
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,μ-u+1:μ+l+1,:) .= f.(data_s)
    elseif μ > u
        fill!(view(data_d,1:(μ-u),:), z)
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        view(data_d,μ-u+1:μ+λ+1,:) .= f.(view(data_s,1:u+λ+1,:))
    elseif λ > l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,1:μ+l+1,:) .= f.(view(data_s,u-μ+1:u+l+1,:))
    else # μ < u && λ < l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        data_d .= f.(view(data_s,u-μ+1:u+λ+1,:))
    end

    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (src,x)::Tuple{AbstractMatrix{T},Number}, ::BandedColumnMajor, ::BandedColumnMajor) where T
    z = f(zero(T),x)
    checkbroadcastband(dest, src, z)

    l,u = bandwidths(src)
    λ,μ = bandwidths(dest)
    m,n = size(src)
    data_d,data_s = bandeddata(dest), bandeddata(src)
    if (l,u) == (λ,μ)
        data_d .= f.(data_s, x)
    elseif μ > u && λ > l
        fill!(view(data_d,1:(μ-u),:), z)
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,μ-u+1:μ+l+1,:) .= f.(data_s,x)
    elseif μ > u
        fill!(view(data_d,1:(μ-u),:), z)
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        view(data_d,μ-u+1:μ+λ+1,:) .= f.(view(data_s,1:u+λ+1,:),x)
    elseif λ > l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,1:μ+l+1,:) .= f.(view(data_s,u-μ+1:u+l+1,:),x)
    else # μ < u && λ < l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        data_d .= f.(view(data_s,u-μ+1:u+λ+1,:),x)
    end

    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (x, src)::Tuple{Number,AbstractMatrix{T}}, ::BandedColumnMajor, ::BandedColumnMajor) where T
    z = f(x, zero(T))
    checkbroadcastband(dest, src, z)

    l,u = bandwidths(src)
    λ,μ = bandwidths(dest)
    m,n = size(src)
    data_d,data_s = bandeddata(dest), bandeddata(src)
    if (l,u) == (λ,μ)
        data_d .= f.(data_s, x)
    elseif μ > u && λ > l
        fill!(view(data_d,1:(μ-u),:), z)
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,μ-u+1:μ+l+1,:) .= f.(x, data_s)
    elseif μ > u
        fill!(view(data_d,1:(μ-u),:), z)
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        view(data_d,μ-u+1:μ+λ+1,:) .= f.(x, view(data_s,1:u+λ+1,:))
    elseif λ > l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,1:μ+l+1,:) .= f.(x, view(data_s,u-μ+1:u+l+1,:))
    else # μ < u && λ < l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        data_d .= f.(x, view(data_s,u-μ+1:u+λ+1,:))
    end

    dest
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{<:AbstractMatrix}})
    (A,) = bc.args
    _banded_broadcast!(dest, bc.f, A, MemoryLayout(dest), MemoryLayout(A))
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{<:AbstractMatrix,<:Number}})
    (A,x) = bc.args
    _banded_broadcast!(dest, bc.f, (A, x), MemoryLayout(dest), MemoryLayout(A))
end


function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{<:Number,<:AbstractMatrix}})
    (x,A) = bc.args
    _banded_broadcast!(dest, bc.f, (x,A), MemoryLayout(dest), MemoryLayout(A))
end


function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{<:AbstractMatrix,<:AbstractMatrix}})
    _banded_broadcast!(dest, bc.f, bc.args, MemoryLayout(dest), MemoryLayout.(bc.args))
end

_bandwidths(::Number) = (-720,-720)
_bandwidths(A) = bandwidths(A)

_band_eval_args() = ()
_band_eval_args(a::Number, b...) = (a, _band_eval_args(b...)...)
_band_eval_args(a::AbstractMatrix{T}, b...) where T = (zero(T), _band_eval_args(b...)...)
_band_eval_args(a::Broadcasted, b...) = (zero(mapreduce(eltype, promote_type, a.args)), _band_eval_args(b...)...)

function bandwidths(bc::Broadcasted{BandedStyle})
    if iszero(bc.f(_band_eval_args(bc.args...)...))
        max.(_bandwidths.(bc.args)...)
    else
        (a,b) = size(bc)
        (a-1,b-1)
    end
end

similar(bc::Broadcasted{BandedStyle}, ::Type{T}) where T = BandedMatrix{T}(undef, size(bc), _bandwidths(bc))



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


# ##
# # axpy!
# ##
#
# # these are the routines of the banded interface of other AbstractMatrices
# banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix) = _banded_axpy!(a, X, Y, MemoryLayout(X), MemoryLayout(Y))
# _banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, ::BandedColumnMajor, ::BandedColumnMajor) =
#     banded_generic_axpy!(a, X, Y)
# _banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, notbandedX, notbandedY) =
#     banded_dense_axpy!(a, X, Y)
#
# # additions and subtractions
# @propagate_inbounds function banded_generic_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
#     n,m = size(X)
#     if (n,m) ≠ size(Y)
#         throw(BoundsError())
#     end
#     Xl, Xu = bandwidths(X)
#     Yl, Yu = bandwidths(Y)
#
#     @boundscheck if Xl > Yl
#         # test that all entries are zero in extra bands
#         for j=1:size(X,2),k=max(1,j+Yl+1):min(j+Xl,n)
#             if inbands_getindex(X, k, j) ≠ 0
#                 throw(BandError(X, (k,j)))
#             end
#         end
#     end
#     @boundscheck if Xu > Yu
#         # test that all entries are zero in extra bands
#         for j=1:size(X,2),k=max(1,j-Xu):min(j-Yu-1,n)
#             if inbands_getindex(X, k, j) ≠ 0
#                 throw(BandError(X, (k,j)))
#             end
#         end
#     end
#
#     l = min(Xl,Yl)
#     u = min(Xu,Yu)
#
#     @inbounds for j=1:m,k=max(1,j-u):min(n,j+l)
#         inbands_setindex!(Y, a*inbands_getindex(X,k,j) + inbands_getindex(Y,k,j) ,k, j)
#     end
#     Y
# end
#
# function banded_dense_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
#     if size(X) != size(Y)
#         throw(DimensionMismatch("+"))
#     end
#     @inbounds for j=1:size(X,2),k=colrange(X,j)
#         Y[k,j] += a*inbands_getindex(X,k,j)
#     end
#     Y
# end
#
#
# function copyto!(dest::AbstractArray{T}, bc::Broadcasted{BandedStyle, <:Any, typeof(+),
#                                                             <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}) where T
#     A,B = bc.args
#     if dest ≡ B
#         banded_axpy!(one(T), A, dest)
#     elseif dest ≡ A
#         banded_axpy!(one(T), B, dest)
#     else
#         banded_copyto!(dest, B)
#         banded_axpy!(one(T), A, dest)
#     end
# end
#
# function similar(bc::Broadcasted{BandedStyle, <:Any, typeof(+), <:Tuple{<:AbstractMatrix,<:AbstractMatrix}}, ::Type{T}) where T
#     A,B = bc.args
#     n,m = size(A)
#     (n,m) == size(B) || throw(DimensionMismatch())
#     Al,Au = bandwidths(A)
#     Bl,Bu = bandwidths(B)
#     similar(A, T, n, m, max(Al,Bl), max(Au,Bu))
# end
#
#
# function copyto!(dest::AbstractArray{T}, bc::Broadcasted{BandedStyle, <:Any, typeof(+),
#                                                         <:Tuple{<:Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
#                                                         <:AbstractMatrix}}) where T
#     αA,B = bc.args
#     α,A = αA.args
#     dest ≡ B || banded_copyto!(dest, B)
#     banded_axpy!(α, A, dest)
# end
#
# function similar(bc::Broadcasted{BandedStyle, <:Any, typeof(+),
#                         <:Tuple{<:Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}},
#                         <:AbstractMatrix}}, ::Type{T}) where T
#     αA,B = bc.args
#     α,A = αA.args
#     n,m = size(A)
#     (n,m) == size(B) || throw(DimensionMismatch())
#     Al,Au = bandwidths(A)
#     Bl,Bu = bandwidths(B)
#     similar(A, T, n, m, max(Al,Bl), max(Au,Bu))
# end
