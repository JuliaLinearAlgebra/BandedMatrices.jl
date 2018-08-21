
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


mul!(dest::AbstractVector, A::AbstractBandedMatrix, x::AbstractVector) =
    (dest .= Mul(A,x))

mul!(dest::AbstractMatrix, A::AbstractBandedMatrix, x::AbstractMatrix) =
    (dest .= Mul(A,x))
mul!(dest::AbstractMatrix, A::AbstractBandedMatrix, x::AbstractBandedMatrix) =
    (dest .= Mul(A,x))
mul!(dest::AbstractMatrix, A::AbstractBandedMatrix, x::Adjoint{<:Any,<:AbstractMatrix}) =
    (dest .= Mul(A,x))

mul!(dest::AbstractVector, A::Adjoint{<:Any,<:AbstractBandedMatrix}, b::AbstractVector) =
    (dest .= Mul(A, b))
mul!(dest::AbstractVector, A::Transpose{<:Any,<:AbstractBandedMatrix}, b::AbstractVector) =
    (dest .= Mul(A, b))

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

@blasmatvec BandedColumnMajor

@inline blasmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α, β,
              ::AbstractStridedLayout, ::BandedColumnMajor, ::AbstractStridedLayout) =
    _banded_gbmv!('N', α, A, x, β, y)

@blasmatvec BandedRowMajor

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::BandedRowMajor, ::AbstractStridedLayout) where T<:BlasFloat =
    _banded_gbmv!('T', α, transpose(A), x, β, y)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::ConjLayout{BandedRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex =
    _banded_gbmv!('C', α, A', x, β, y)


##
# Non-BLAS mat vec
##



@inline function _copyto!(_, c::AbstractVector{T},
         bc::BMixedMatVec{<:Any, <:BandedColumnMajor}) where T
    (M,) = bc.args
    A,b = M.A, M.B
    fill!(c, zero(T))
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[k] += inbands_getindex(A,k,j)*b[j]
    end
    c
end


@inline function _copyto!(_, c::AbstractVector{T},
         bc::BMixedMatVec{<:Any, <:BandedRowMajor}) where T
    (M,) = bc.args
    At,b = M.A, M.B
    A = transpose(At)
    fill!(c, zero(T))
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[j] += transpose(inbands_getindex(A,k,j))*b[k]
    end
    c
end

@inline function _copyto!(_, c::AbstractVector{T},
         bc::BMixedMatVec{<:Any, <:ConjLayout{<:BandedRowMajor}}) where T
    (M,) = bc.args
    Ac,b = M.A, M.B
    A = Ac'
    fill!(c, zero(T))
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[j] += inbands_getindex(A,k,j)'*b[k]
    end
    c
end


## Banded * Banded
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
         bc::BMixedMatMat{<:Any,<:ConjOrBandedLayout,<:ConjOrBandedLayout})
     (M,) = bc.args
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix,
         bc::BMixedMatMat{<:Any,<:ConjOrBandedLayout})
     (M,) = bc.args
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix,
         bc::BMixedMatMat{<:Any,<:Any,<:ConjOrBandedLayout})
     (M,) = bc.args
     A, B = M.A, M.B
     banded_mul!(C, A, B)
end




#
# function generally_banded_matvecmul!(c::AbstractVector{T}, tA::Val, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V}
#     m, n = _size(tA, A)
#     if length(c) ≠ m || length(b) ≠ n
#         throw(DimensionMismatch("*"))
#     end
#
#     l, u = _bandwidths(tA, A)
#     if -l > u
#         # no bands
#         fill!(c, zero(T))
#     elseif l < 0
#         banded_matvecmul!(c, tA, _view(tA, A, :, 1-l:n), view(b, 1-l:n))
#     elseif u < 0
#         c[1:-u] .= zero(T)
#         banded_matvecmul!(view(c, 1-u:m), tA, _view(tA, A, 1-u:m, :), b)
#     else
#         positively_banded_matvecmul!(c, tA, A, b)
#     end
#     c
# end
