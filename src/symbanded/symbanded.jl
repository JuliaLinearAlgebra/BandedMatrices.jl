#
# Represent a symmetric banded matrix
# [ a_11 a_12 a_13
#   a_12 a_22 a_23 a_24
#   a_13 a_23 a_33 a_34
#        a_24 a_34 a_44  ]
# ordering the data like  (columns first)
#       [ *      *      a_13   a_24
#         *      a_12   a_23   a_34
#         a_11   a_22   a_33   a_44 ]
###




@inline function inbands_getindex(A::Symmetric{T, <:BandedMatrix}, k::Integer, j::Integer) where T
    P = parent(A)
    l,u = bandwidths(P)
    if -l ≤ abs(k-j) ≤ u
        parent(A).data[bandwidth(A) - abs(k-j) + 1, max(k,j)]
    else
        zero(T)
    end
end

# this is a hack but is much faster than default
function getindex(S::Symmetric{<:Any, <:BandedMatrix}, kr::AbstractUnitRange, jr::AbstractUnitRange)
    A = parent(S)
    m = max(last(kr), last(jr))
    B = if S.uplo == 'U'
        BandedMatrix(A[1:m,1:m],(0, bandwidth(A,2)))
    else
        BandedMatrix(A[1:m,1:m],(bandwidth(A,1),0))
    end
    ret = B + transpose(B)
    rdiv!(view(ret, band(0)), 2)
    ret[kr, jr]
end

symmetriclayout(::ML) where ML<:BandedColumns = SymmetricLayout{ML}()
symmetriclayout(::ML) where ML<:BandedRows = SymmetricLayout{ML}()

hermitianlayout(::Type{<:Complex}, ::ML) where ML<:BandedColumns = HermitianLayout{ML}()
hermitianlayout(::Type{<:Real}, ::ML) where ML<:BandedColumns = SymmetricLayout{ML}()
hermitianlayout(::Type{<:Complex}, ::ML) where ML<:BandedRows = HermitianLayout{ML}()
hermitianlayout(::Type{<:Real}, ::ML) where ML<:BandedRows = SymmetricLayout{ML}()

sublayout(::SymmetricLayout{<:AbstractBandedLayout}, ::Type{<:NTuple{2,AbstractUnitRange}}) = BandedLayout()
sublayout(::HermitianLayout{<:AbstractBandedLayout}, ::Type{<:NTuple{2,AbstractUnitRange}}) = BandedLayout()


isbanded(A::HermOrSym) = isbanded(parent(A))

bandwidth(A::HermOrSym) = ifelse(symmetricuplo(A) == 'U', bandwidth(parent(A),2), bandwidth(parent(A),1))
bandwidths(A::HermOrSym) = (bandwidth(A), bandwidth(A))


for (f,g) in [(:symbandeddata, :symmetricdata), (:hermbandeddata, :hermitiandata)]
    @eval function $f(A)
        B = $g(A)
        l,u = bandwidths(B)
        D = bandeddata(B)
        if symmetricuplo(A) == 'U'
            view(D, 1:u+1, :)
        else
            m = size(D,1)
            view(D, u+1:u+l+1, :)
        end
    end
end

banded_sbmv!(uplo, α::T, A::AbstractMatrix{T}, x::AbstractVector{T}, β::T, y::AbstractVector{T}) where {T<:BlasFloat} =
    BLAS.sbmv!(uplo, bandwidth(A), α, symbandeddata(A), x, β, y)


@inline function _banded_sbmv!(tA, α, A, x, β, y)
    if x ≡ y
        banded_sbmv!(tA, α, A, copy(x), β, y)
    else
        banded_sbmv!(tA, α, A, x, β, y)
    end
end


function materialize!(M::BlasMatMulVecAdd{<:SymmetricLayout{<:BandedColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l = bandwidth(A)
    l ≥ 0 || return lmul!(β, y)
    _banded_sbmv!(symmetricuplo(A), α, A, x, β, y)
end

banded_hbmv!(uplo, α::T, A::AbstractMatrix{T}, x::AbstractVector{T}, β::T, y::AbstractVector{T}) where {T<:BlasFloat} =
    BLAS.hbmv!(uplo, bandwidth(A), α, hermbandeddata(A), x, β, y)


@inline function _banded_hbmv!(tA, α, A, x, β, y)
    if x ≡ y
        banded_hbmv!(tA, α, A, copy(x), β, y)
    else
        banded_hbmv!(tA, α, A, x, β, y)
    end
end

function materialize!(M::BlasMatMulVecAdd{<:HermitianLayout{<:BandedColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, A, x, β, y = M.α, M.A, M.B, M.β, M.C
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l = bandwidth(A)
    l ≥ 0 || return lmul!(β, y)
    _banded_hbmv!(symmetricuplo(A), α, A, x, β, y)
end

function copyto!(A::Symmetric{<:Number,<:BandedMatrix}, B::Symmetric{<:Number,<:BandedMatrix})
    size(A) == size(B) || throw(ArgumentError("sizes of A and B must match"))
    bandwidth(A) >= bandwidth(B) || throw(ArgumentError("bandwidth of A must exceed that of B"))
    A .= zero(eltype(A))
    SAuplo = symmetricuplo(A)
    SBuplo = symmetricuplo(B)
    if SAuplo == SBuplo
        ASdata = symbandeddata(A)
        BSdata = symbandeddata(B)
        if SAuplo == 'L'
            ASdata[axes(BSdata)...] = BSdata
        else
            nrowsskip = size(ASdata,1) - size(BSdata,1)
            ASdata[axes(BSdata,1) .+ nrowsskip, axes(BSdata,2)] = BSdata
        end
    else
        Ap = parent(A)
        Bp = parent(B)
        if SAuplo == 'L'
            LowerTriangular(Ap) .= LowerTriangular(transpose(Bp))
        else
            UpperTriangular(Ap) .= UpperTriangular(transpose(Bp))
        end
    end
    return A
end

function _copy_bandedsym(A, B)
    if bandwidth(A) >= bandwidth(B)
        copy(A)
    else
        copyto!(similar(B), A)
    end
end

function eigvals(A::Symmetric{<:Any,<:BandedMatrix}, B::Symmetric{<:Any,<:BandedMatrix})
    AA = _copy_bandedsym(A, B)
    eigvals!(AA, copy(B))
end
