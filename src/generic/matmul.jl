bandwidths(M::MulAdd) = min.(_bnds(M), prodbandwidths(M.A,M.B))
bandwidths(M::Lmul) = min.(_bnds(M), prodbandwidths(M.A,M.B))
bandwidths(M::Rmul) = min.(_bnds(M), prodbandwidths(M.A,M.B))
bandwidths(M::ArrayLayouts.Mul) = min.(_bnds(M), prodbandwidths(M.A,M.B))
similar(M::MulAdd{<:DiagonalLayout,<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))
similar(M::MulAdd{<:AbstractBandedLayout,<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))
similar(M::MulAdd{<:AbstractBandedLayout,<:DiagonalLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))
similar(M::MulAdd{<:SymmetricLayout{<:AbstractBandedLayout},<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))
similar(M::MulAdd{<:HermitianLayout{<:AbstractBandedLayout},<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where T =
    BandedMatrix{T}(undef, axes, bandwidths(M))
similar(M::MulAdd{<:TriangularLayout{uplo,trans,<:AbstractBandedLayout},<:AbstractBandedLayout}, ::Type{T}, axes::NTuple{2,OneTo{Int}}) where {uplo,trans,T} =
    BandedMatrix{T}(undef, axes, bandwidths(M))
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





function _banded_muladd!(α::T, A, x::AbstractVector, β, y) where T
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l, u = bandwidths(A)
    if -l > u # no bands
        _fill_lmul!(β, y)
    elseif l < 0
        _banded_muladd!(α, view(A, :, 1-l:n), view(x, 1-l:n), β, y)
    elseif u < 0
        y[1:-u] .= zero(T)
        _banded_muladd!(α, view(A, 1-u:m, :), x, β, view(y, 1-u:m))
        y
    else
        _banded_gbmv!('N', α, A, x, β, y)
    end
end

materialize!(M::BlasMatMulVecAdd{<:BandedColumnMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasFloat =
    _banded_muladd!(M.α, M.A, M.B, M.β, M.C)

function materialize!(M::BlasMatMulVecAdd{<:BandedRowMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasFloat
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    At = transpose(A)
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
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

function materialize!(M::BlasMatMulVecAdd{<:ConjLayout{<:BandedRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,T}) where T<:BlasComplex
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    Ac = A'
    m, n = size(A)
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
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

@inline function materialize!(M::MatMulVecAdd{<:AbstractBandedLayout})
    checkdimensions(M)
    α,A,B,β,C = M.α,M.A,M.B,M.β,M.C
    _fill_lmul!(β, C)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        C[k] += α*inbands_getindex(A,k,j)*B[j]
    end
    C
end

@inline function materialize!(M::MatMulVecAdd{<:BandedRowMajor})
    checkdimensions(M)
    α,At,B,β,C = M.α,M.A,M.B,M.β,M.C
    A = transpose(At)
    _fill_lmul!(β, C)

    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        C[j] +=  α*transpose(inbands_getindex(A,k,j))*B[k]
    end
    C
end

@inline function materialize!(M::MatMulVecAdd{<:ConjLayout{<:BandedRowMajor}})
    checkdimensions(M)
    α,Ac,B,β,C = M.α,M.A,M.B,M.β,M.C
    A = Ac'
    _fill_lmul!(β, C)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        C[j] += α*inbands_getindex(A,k,j)'*B[k]
    end
    C
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

function _banded_muladd!(α::T, A, B::AbstractMatrix, β, C) where T
    Am, An = size(A)
    Bm, Bn = size(B)
    if An != Bm || size(C, 1) != Am || size(C, 2) != Bn
        throw(DimensionMismatch("*"))
    end

    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)

    gbmm!('N', 'N', α, A, B, β, C)

    C
end

materialize!(M::BlasMatMulMatAdd{<:AbstractBandedLayout,<:AbstractBandedLayout,<:BandedColumnMajor}) =
    materialize!(MulAdd(M.α, convert(DefaultBandedMatrix,M.A), convert(DefaultBandedMatrix,M.B), M.β, M.C))

materialize!(M::BlasMatMulMatAdd{<:BandedColumnMajor,<:BandedColumnMajor,<:BandedColumnMajor}) =
    _banded_muladd!(M.α, M.A, M.B, M.β, M.C)


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



###
# Special Fill Diagonal
####

function materialize!(M::MatMulMatAdd{<:DiagonalLayout{<:AbstractFillLayout},<:AbstractBandedLayout})
    checkdimensions(M)
    M.C .= (M.α * getindex_value(M.A.diag)) .* M.B .+ M.β .* M.C
    M.C
end

function materialize!(M::MatMulMatAdd{<:AbstractBandedLayout,<:DiagonalLayout{<:AbstractFillLayout}})
    checkdimensions(M)
    M.C .= (M.α * getindex_value(M.B.diag)) .* M.A .+ M.β .* M.C
    M.C
end
