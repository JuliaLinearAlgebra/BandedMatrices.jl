BroadcastStyle(M::ArrayMulArrayStyle, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ArrayMulArrayStyle) = M

@lazymul AbstractBandedMatrix

bandwidths(M::Mul) = prodbandwidths(M.args...)

similar(M::MatMulMat{<:AbstractBandedLayout,<:AbstractBandedLayout}, ::Type{T}) where T =
    similar(M, T, axes(M))

similar(M::MatMulMat{<:AbstractBandedLayout,<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))


### Symmetric

for Lay in (:SymmetricLayout, :HermitianLayout)
    @eval begin
        similar(M::MatMulMat{<:$Lay{<:AbstractBandedLayout},<:$Lay{<:AbstractBandedLayout}}, ::Type{T}) where T =
            BandedMatrix{T}(undef, size(M), bandwidths(M))
        similar(M::MatMulMat{<:$Lay{<:AbstractBandedLayout},<:AbstractBandedLayout}, ::Type{T}) where T =
            BandedMatrix{T}(undef, size(M), bandwidths(M))
        similar(M::MatMulMat{<:AbstractBandedLayout,<:$Lay{<:AbstractBandedLayout}}, ::Type{T}) where T =
            BandedMatrix{T}(undef, size(M), bandwidths(M))
    end
end

### Triangular

similar(M::MatMulMat{<:TriangularLayout{uplo1,unit1,<:AbstractBandedLayout},
                     <:TriangularLayout{uplo2,unit2,<:AbstractBandedLayout}}, ::Type{T}) where {uplo1,uplo2,unit1,unit2,T} =
    BandedMatrix{T}(undef, size(M), bandwidths(M))
similar(M::MatMulMat{<:TriangularLayout{uplo,unit,<:AbstractBandedLayout},<:AbstractBandedLayout}, ::Type{T}) where {uplo,unit,T} =
    BandedMatrix{T}(undef, size(M), bandwidths(M))
similar(M::MatMulMat{<:AbstractBandedLayout,<:TriangularLayout{uplo,unit,<:AbstractBandedLayout}}, ::Type{T}) where {uplo,unit,T} =
    BandedMatrix{T}(undef, size(M), bandwidths(M))


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





function materialize!(M::BlasMatMulVec{<:BandedColumnMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasFloat
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
        _fill_lmul!(β, y)
    elseif l < 0
        materialize!(MulAdd(α, view(A, :, 1-l:n), view(x, 1-l:n), β, y))
    elseif u < 0
        y[1:-u] .= zero(T)
        materialize!(MulAdd(α, view(A, 1-u:m, :), x, β, view(y, 1-u:m)))
        y
    else
        _banded_gbmv!('N', α, A, x, β, y)
    end
end

function materialize!(M::BlasMatMulVec{<:BandedRowMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasFloat
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    At = transpose(A)
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
      _fill_lmul!(β, y)
    elseif l < 0
      materialize!(MulAdd(α, transpose(view(At, 1-l:n, :,)), view(x, 1-l:n), β, y))
    elseif u < 0
      y[1:-u] .= zero(T)
      materialize!(MulAdd(α, transpose(view(At, :, 1-u:m)), x, β, view(y, 1-u:m)))
      y
    else
      _banded_gbmv!('T', α, At, x, β, y)
    end
end

function materialize!(M::BlasMatMulVec{<:ConjLayout{<:BandedRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasComplex
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    Ac = A'
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
        _fill_lmul!(β, y)
    elseif l < 0
        materialize!(MulAdd(α, view(Ac, 1-l:n, :,)', view(x, 1-l:n), β, y))
    elseif u < 0
        y[1:-u] .= zero(T)
        materialize!(MulAdd(α, view(Ac, :, 1-u:m)', x, β, view(y, 1-u:m)))
        y
    else
    _banded_gbmv!('C', α, Ac, x, β, y)
    end
end



##
# Non-BLAS mat vec
##


@inline function _copyto!(_, C::AbstractVector{T}, M::MatMulVec{<:BandedColumnMajor,<:AbstractStridedLayout,T,T}) where T<:BlasFloat
    A,B = M.args
    materialize!(MulAdd(one(T), A, B, zero(T), C))
end

@inline function _copyto!(_, C::AbstractVector{T}, M::MatMulVec{<:BandedRowMajor,<:AbstractStridedLayout,T,T}) where T<:BlasFloat
    A,B = M.args
    materialize!(MulAdd(one(T), A, B, zero(T), C))
end

@inline function _copyto!(_, C::AbstractVector{T}, M::MatMulVec{<:ConjLayout{<:BandedRowMajor},<:AbstractStridedLayout,T,T}) where T<:BlasFloat
    A,B = M.args
    materialize!(MulAdd(one(T), A, B, zero(T), C))
end


@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:AbstractBandedLayout})
    A,b = M.args
    for k = 1:length(c)
        c[k] = zero(eltype(A)) * b[1]
    end
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[k] += inbands_getindex(A,k,j)*b[j]
    end
    c
end

@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:BandedRowMajor})
    At,b = M.args
    A = transpose(At)
    c .= zero.(c)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[j] += transpose(inbands_getindex(A,k,j))*b[k]
    end
    c
end

@inline function _copyto!(_, c::AbstractVector, M::MatMulVec{<:ConjLayout{<:BandedRowMajor}})
    Ac,b = M.args
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
        for k = max(j - C̃u,1):min(j-Cu-1,Am)
            inbands_setindex!(C,zero(T),k,j)
        end
        for k = max(j + Cl+1,1):min(j+C̃l,Am)
            inbands_setindex!(C,zero(T),k,j)
        end
    end

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):min(j+Cl, Am)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,ν,j)
        end
        setindex!(C,tmp,k,j)
    end
    C
end

const ConjOrBandedLayout = Union{AbstractBandedLayout,ConjLayout{<:AbstractBandedLayout}}
const ConjOrBandedColumnMajor = Union{<:BandedColumnMajor,ConjLayout{<:BandedColumnMajor}}

@inline function _copyto!(_, C::AbstractMatrix{T}, M::MatMulMat{<:ConjOrBandedColumnMajor,<:ConjOrBandedColumnMajor,T,T}) where T<:BlasFloat
    A,B = M.args
    materialize!(MulAdd(one(T), A, B, zero(T), C))
end

@inline function _copyto!(_, C::AbstractMatrix, M::MatMulMat{<:ConjOrBandedLayout,<:ConjOrBandedLayout})
     A, B = M.args
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix, M::MatMulMat{<:ConjOrBandedLayout})
     A, B = M.args
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix, M::MatMulMat{<:Any,<:ConjOrBandedLayout})
     A, B = M.args
     banded_mul!(C, A, B)
end

@inline function _copyto!(_, C::AbstractMatrix, M::MatMulMat{<:TriangularLayout,<:ConjOrBandedLayout})
     A, B = M.args
     banded_mul!(C, A, B)
end

function materialize!(M::BlasMatMulMat{<:BandedColumnMajor,<:BandedColumnMajor,<:BandedColumnMajor,T}) where T<:BlasFloat
    α, A, B, β, C = M.α, M.A, M.B, M.β, M.C

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
        materialize!(MulAdd(α, view(A, :, 1-Al:An), view(B, 1-Al:An, :), β, C))
    elseif Au < 0
        _fill_lmul!(β, @views(C[1:-Au,:]))
        materialize!(MulAdd(α, view(A, 1-Au:Am,:), B, β, view(C, 1-Au:Am,:)))
    elseif Bl < 0
        _fill_lmul!(β, @views(C[:, 1:-Bl]))
        materialize!(MulAdd(α, A, view(B, :, 1-Bl:Bn), β, view(C, :, 1-Bl:Bn)))
    elseif Bu < 0
        _fill_lmul!(β, @views(C[:, max(1,Am+Bu-1):Bn]))
        materialize!(MulAdd(α, view(A, :, 1-Bu:Bm), view(B, 1-Bu:Bm, :), β, C))
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
