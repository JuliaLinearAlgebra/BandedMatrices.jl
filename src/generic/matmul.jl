BroadcastStyle(M::ArrayMulArrayStyle, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ArrayMulArrayStyle) = M

@blasmatvec BandedColumnMajor
@blasmatvec BandedRowMajor
@blasmatmat BandedColumnMajor BandedColumnMajor BandedColumnMajor
@lazymul AbstractBandedMatrix


###
# For <: AbstractBandedMatrix, we can use Base routines like `mul!`
# For something only conforming to the banded matrix interface, we
# need to use LazyArrays style: `b .= Mul(A,x)`
# note only BLAS types have fast `c .= α.*Mul(A,x) .+ β.*b`
###

function *(A::AbstractBandedMatrix{T}, B::AbstractBandedMatrix{V}) where {T, V}
    n, m = size(A,1), size(B,2)
    Y = BandedMatrix{promote_type(T,V)}(undef, n, m, prodbandwidths(A, B)...)
    Y .= Mul(A, B)
end
function *(A::AbstractBandedMatrix{T}, B::AdjOrTrans{V,<:AbstractBandedMatrix{V}}) where {T, V}
    n, m = size(A,1), size(B,2)
    Y = BandedMatrix{promote_type(T,V)}(undef, n, m, prodbandwidths(A, B)...)
    Y .= Mul(A, B)
end
function *(A::AdjOrTrans{T,<:AbstractBandedMatrix{T}}, B::AbstractBandedMatrix{V}) where {T, V}
    n, m = size(A,1), size(B,2)
    Y = BandedMatrix{promote_type(T,V)}(undef, n, m, prodbandwidths(A, B)...)
    Y .= Mul(A, B)
end
function *(A::AdjOrTrans{T,<:AbstractBandedMatrix{T}}, B::AdjOrTrans{V,<:AbstractBandedMatrix{V}}) where {T, V}
    n, m = size(A,1), size(B,2)
    Y = BandedMatrix{promote_type(T,V)}(undef, n, m, prodbandwidths(A, B)...)
    Y .= Mul(A, B)
end



##
# BLAS routines
##

# make copy to make sure always works

banded_gbmv!(tA, α, A, x, β, y) =
    BLAS.gbmv!(tA, size(A,1), bandwidth(A,1), bandwidth(A,2),
                α, bandeddata(A), x, β, y)


@inline function _banded_gbmv!(tA, α, A, x, β, y)
    if x ≡ y
        banded_gbmv!(tA, α, A, copy(x), β, y)
    else
        banded_gbmv!(tA, α, A, x, β, y)
    end
end



function blasmul!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α::T, β::T,
                ::AbstractStridedLayout, ::BandedColumnMajor, ::AbstractStridedLayout) where T<:BlasFloat
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
        lmul!(β, y)
    elseif l < 0
        blasmul!(y, view(A, :, 1-l:n), view(x, 1-l:n), α, β)
    elseif u < 0
        y[1:-u] .= zero(T)
        blasmul!(view(y, 1-u:m), view(A, 1-u:m, :), x, α, β)
        y
    else
        _banded_gbmv!('N', α, A, x, β, y)
    end
end

function blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
                   ::AbstractStridedLayout, ::BandedRowMajor, ::AbstractStridedLayout) where T<:BlasFloat
    At = transpose(A)
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
      _fill_lmul!(β, y)
    elseif l < 0
      blasmul!(y, transpose(view(At, 1-l:n, :,)), view(x, 1-l:n), α, β)
    elseif u < 0
      y[1:-u] .= zero(T)
      blasmul!(view(y, 1-u:m), transpose(view(At, :, 1-u:m)), x, α, β)
      y
    else
      _banded_gbmv!('T', α, At, x, β, y)
    end
end

function blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
                ::AbstractStridedLayout, ::ConjLayout{BandedRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex
    Ac = A'
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
        _fill_lmul!(β, y)
    elseif l < 0
        blasmul!(y, view(Ac, 1-l:n, :,)', view(x, 1-l:n), α, β)
    elseif u < 0
        y[1:-u] .= zero(T)
        blasmul!(view(y, 1-u:m), view(Ac, :, 1-u:m)', x, α, β)
        y
    else
    _banded_gbmv!('C', α, Ac, x, β, y)
    end
end



##
# Non-BLAS mat vec
##



@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:Any, <:AbstractBandedLayout})
    A,b = M.A, M.B
    for k = 1:length(c)
        c[k] = zero(eltype(A)) * b[1]
    end
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[k] += inbands_getindex(A,k,j)*b[j]
    end
    c
end

@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:Any, <:BandedRowMajor})
    At,b = M.A, M.B
    A = transpose(At)
    c .= zero.(c)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[j] += transpose(inbands_getindex(A,k,j))*b[k]
    end
    c
end

@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:Any, <:ConjLayout{<:BandedRowMajor}})
    Ac,b = M.A, M.B
    A = Ac'
    c .= zero.(c)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[j] += inbands_getindex(A,k,j)'*b[k]
    end
    c
end


####
# Matrix * Matrix
####
function banded_mul!(C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix) where T
    Am, An = size(A)
    Bm, Bn = size(B)
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    Cl,Cu = prodbandwidths(A, B)
    size(C) == (Am,Bn) || throw(DimensionMismatch())
    C̃l,C̃u = bandwidths(C)
    # set extra bands to zero
    @inbounds for j = 1:Bn
        for k = max(j - C̃u,1):max(min(j-Cu-1,Am),0)
            inbands_setindex!(C,zero(T),k,j)
        end
        for k = max(j + Cl+1,1):max(min(j+C̃l,Am),0)
            inbands_setindex!(C,zero(T),k,j)
        end
    end

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,ν,j)
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

const ConjOrBandedLayout = Union{AbstractBandedLayout,ConjLayout{<:AbstractBandedLayout}}

@inline function _copyto!(_, C::AbstractMatrix,
         M::MatMulMat{<:Any,<:ConjOrBandedLayout,<:ConjOrBandedLayout})
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix,
         M::MatMulMat{<:Any,<:ConjOrBandedLayout})
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix,
         M::MatMulMat{<:Any,<:Any,<:ConjOrBandedLayout})
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end



function blasmul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, α::T, β::T,
                ::BandedColumnMajor, ::BandedColumnMajor, ::BandedColumnMajor) where T<:BlasFloat
    Am, An = size(A)
    Bm, Bn = size(B)
    if An != Bm || size(C, 1) != Am || size(C, 2) != Bn
        throw(DimensionMismatch("*"))
    end

    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)

    if (-Al > Au) || (-Bl > Bu)   # A or B has empty bands
        fill!(C, zero(T))
    elseif Al < 0
        _fill_lmul!(β, @views(C[max(1,Bn+Al-1):Am, :]))
        blasmul!(C, view(A, :, 1-Al:An), view(B, 1-Al:An, :), α, β)
    elseif Au < 0
        _fill_lmul!(β, @views(C[1:-Au,:]))
        blasmul!(view(C, 1-Au:Am,:), view(A, 1-Au:Am,:), B, α, β)
    elseif Bl < 0
        _fill_lmul!(β, @views(C[:, 1:-Bl]))
        blasmul!(view(C, :, 1-Bl:Bn), A, view(B, :, 1-Bl:Bn), α, β)
    elseif Bu < 0
        _fill_lmul!(β, @views(C[:, max(1,Am+Bu-1):Bn]))
        blasmul!(C, view(A, :, 1-Bu:Bm), view(B, 1-Bu:Bm, :), α, β)
    else
        gbmm!('N', 'N', α, A, B, β, C)
    end
    C
end


# function generally_banded_matmatmul!(C::AbstractMatrix{T}, tA::Val, tB::Val, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
#     Am, An = _size(tA, A)
#     Bm, Bn = _size(tB, B)
#     if An != Bm || size(C, 1) != Am || size(C, 2) != Bn
#         throw(DimensionMismatch("*"))
#     end
#     # TODO: checkbandmatch
#
#     Al, Au = _bandwidths(tA, A)
#     Bl, Bu = _bandwidths(tB, B)
#
#     if (-Al > Au) || (-Bl > Bu)   # A or B has empty bands
#         fill!(C, zero(T))
#     elseif Al < 0
#         C[max(1,Bn+Al-1):Am, :] .= zero(T)
#         banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Al:An), _view(tB, B, 1-Al:An, :))
#     elseif Au < 0
#         C[1:-Au,:] .= zero(T)
#         banded_matmatmul!(view(C, 1-Au:Am,:), tA, tB, _view(tA, A, 1-Au:Am,:), B)
#     elseif Bl < 0
#         C[:, 1:-Bl] .= zero(T)
#         banded_matmatmul!(view(C, :, 1-Bl:Bn), tA, tB, A, _view(tB, B, :, 1-Bl:Bn))
#     elseif Bu < 0
#         C[:, max(1,Am+Bu-1):Bn] .= zero(T)
#         banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Bu:Bm), _view(tB, B, 1-Bu:Bm, :))
#     else
#         positively_banded_matmatmul!(C, tA, tB, A, B)
#     end
#     C
# end
