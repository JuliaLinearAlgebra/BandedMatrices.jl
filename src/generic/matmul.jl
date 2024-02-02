bandwidths(M::Union{MulAdd, Lmul, Rmul, ArrayLayouts.Mul}) = min.(_bnds(M), prodbandwidths(M.A,M.B))

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


@inline function _banded_gbmv!(tA, α, A, x, β, y, yzero=false)
    #= Some BLAS implementations throw warnings
    with zero-sized arrays, so we handle
    these cases separately.
    =#
    length(y) == 0 && return y
    if length(x) == 0
        _fill_rmul!(y, β, yzero)
    else
        xc = Base.unalias(y, x)
        banded_gbmv!(tA, α, A, xc, β, y)
    end
    return y
end

function _banded_muladd!(α, A, x::AbstractVector, β, y, yzero)
    m, n = size(A)
    l, u = bandwidths(A)
    if -l > u # no bands
        _fill_rmul!(y, β, yzero)
    elseif l < 0 # with u >= -l > 0, that is, all bands lie above the diagonal
        # E.g. (l,u) = (-1,2)
        # set lview = 0 and uview = u + l >= 0
        _banded_gbmv!('N', α, view(A, :, 1-l:n), view(x, 1-l:n), β, y, yzero)
    elseif u < 0 # with -l <= u < 0, that is, all bands lie below the diagnoal.
        # E.g. (l,u) = (2,-1)
        # set lview = l + u >= 0 and uview = 0
        _fill_rmul!(@view(y[1:-u]), β, yzero)
        _banded_gbmv!('N', α, view(A, 1-u:m, :), x, β, view(y, 1-u:m), yzero)
        y
    else
        _banded_gbmv!('N', α, A, x, β, y, yzero)
    end
end

function materialize!(M::BlasMatMulVecAdd{<:BandedColumnMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat})
    checkdimensions(M)
    _banded_muladd!(M.α, M.A, M.B, M.β, M.C, M.Czero)
end

function _banded_muladd_row!(tA, α, At, x, β, y, yzero=false)
    n, m = size(At)
    u, l = bandwidths(At)
    if -l > u # no bands
        _fill_rmul!(y, β, yzero)
    elseif l < 0
        _banded_gbmv!(tA, α, view(At, 1-l:n, :,), view(x, 1-l:n), β, y, yzero)
    elseif u < 0
        _fill_rmul!(@view(y[1:-u]), β, yzero)
        _banded_gbmv!(tA, α, view(At, :, 1-u:m), x, β, view(y, 1-u:m), yzero)
        y
    else
        _banded_gbmv!(tA, α, At, x, β, y, yzero)
    end
end

function materialize!(M::BlasMatMulVecAdd{<:BandedRowMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat})
    checkdimensions(M)
    α, A, x, β, y, yzero = M.α, M.A, M.B, M.β, M.C, M.Czero
    _banded_muladd_row!('T', α, transpose(A), x, β, y, yzero)
end

function materialize!(M::BlasMatMulVecAdd{<:ConjLayout{<:BandedRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex})
    checkdimensions(M)
    α, A, x, β, y, yzero = M.α, M.A, M.B, M.β, M.C, M.Czero
    _banded_muladd_row!('C', α, A', x, β, y, yzero)
end



##
# Non-BLAS mat vec
##

@inline function materialize!(M::MatMulVecAdd{<:AbstractBandedLayout})
    checkdimensions(M)
    α,A,B,β,C = M.α,M.A,M.B,M.β,M.C
    _fill_rmul!(M, β)
    @inbounds for j = intersect(rowsupport(A), colsupport(B))
        for k = colrange(A,j)
            C[k] += inbands_getindex(A,k,j) * B[j] * α
        end
    end
    C
end

@inline function materialize!(M::MatMulVecAdd{<:BandedRowMajor})
    checkdimensions(M)
    α,At,B,β,C = M.α,M.A,M.B,M.β,M.C
    A = transpose(At)
    _fill_rmul!(M, β)

    @inbounds for j = rowsupport(A)
        for k = intersect(colrange(A,j), colsupport(B))
            C[j] +=  transpose(inbands_getindex(A,k,j)) * B[k] * α
        end
    end
    C
end

@inline function materialize!(M::MatMulVecAdd{<:ConjLayout{<:BandedRowMajor}})
    checkdimensions(M)
    α,Ac,B,β,C = M.α,M.A,M.B,M.β,M.C
    A = Ac'
    _fill_rmul!(M, β)
    @inbounds for j = rowsupport(A)
        for k = intersect(colrange(A,j), colsupport(B))
            C[j] += inbands_getindex(A,k,j)' * B[k] * α
        end
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

function _banded_muladd!(α::T, A, B::AbstractMatrix, β, C, Czero=false) where T
    gbmm!('N', 'N', α, A, B, β, C, Czero)
    C
end

materialize!(M::BlasMatMulMatAdd{<:AbstractBandedLayout,<:AbstractBandedLayout,<:BandedColumnMajor}) =
    materialize!(MulAdd(M.α, convert(DefaultBandedMatrix,M.A), convert(DefaultBandedMatrix,M.B),
        M.β, M.C; Czero = M.Czero))

function materialize!(M::BlasMatMulMatAdd{<:BandedColumnMajor,<:BandedColumnMajor,<:BandedColumnMajor})
    checkdimensions(M)
    _banded_muladd!(M.α, M.A, M.B, M.β, M.C, M.Czero)
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



###
# Special Fill Diagonal
####

function materialize!(M::MatMulMatAdd{<:DiagonalLayout{<:AbstractFillLayout},<:AbstractBandedLayout})
    checkdimensions(M)
    M.C .= getindex_value(M.A.diag) .* M.B .* M.α .+ M.C .* M.β
    M.C
end

function materialize!(M::MatMulMatAdd{<:AbstractBandedLayout,<:DiagonalLayout{<:AbstractFillLayout}})
    checkdimensions(M)
    M.C .= M.A .* getindex_value(M.B.diag) .* M.α .+ M.C .* M.β
    M.C
end

### BandedMatrix * dense matrix

function materialize!(M::MatMulMatAdd{<:BandedColumns, <:AbstractStridedLayout, <:AbstractStridedLayout})
    checkdimensions(M)
    α, β, A, B, C = M.α, M.β, M.A, M.B, M.C

    if iszero(α)
        _fill_rmul!(M, β)
    else
        for (colC, colB) in zip(eachcol(C), eachcol(B))
            mul!(colC, A, colB, α, β)
        end
    end

    return C
end

function materialize!(M::MatMulMatAdd{<:AbstractStridedLayout, <:BandedColumns, <:AbstractStridedLayout})
    checkdimensions(M)
    α, β, A, B, C = M.α, M.β, M.A, M.B, M.C

    if iszero(α)
        _fill_rmul!(M, β)
    else
        for (rowC, rowA) in zip(eachrow(C), eachrow(A))
            mul!(rowC, transpose(B), rowA, α, β)
        end
    end

    return C
end
