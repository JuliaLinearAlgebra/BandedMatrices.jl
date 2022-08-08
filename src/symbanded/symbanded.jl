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


function symbandeddata(A)
    B = symmetricdata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    if symmetricuplo(A) == 'U'
        view(D, 1:u+1, :)
    else
        m = size(D,1)
        view(D, u+1:u+l+1, :)
    end
end

function hermbandeddata(A)
    B = hermitiandata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    if symmetricuplo(A) == 'U'
        view(D, 1:u+1, :)
    else
        m = size(D,1)
        view(D, u+1:u+l+1, :)
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


## eigvals routine


function _tridiagonalize!(A::AbstractMatrix{T}, ::SymmetricLayout{<:BandedColumnMajor}) where T
    n=size(A, 1)
    d = Vector{T}(undef,n)
    e = Vector{T}(undef,n-1)
    Q = Matrix{T}(undef,0,0)
    work = Vector{T}(undef,n)
    sbtrd!('N', symmetricuplo(A), size(A,1), bandwidth(A), symbandeddata(A), d, e, Q, work)
    SymTridiagonal(d,e)
end

tridiagonalize!(A::AbstractMatrix) = _tridiagonalize!(A, MemoryLayout(typeof(A)))

eigvals(A::Symmetric{T,<:BandedMatrix{T}}) where T<:Base.IEEEFloat = eigvals!(copy(A))
eigvals(A::Symmetric{T,<:BandedMatrix{T}}) where T<:Real = eigvals!(tridiagonalize(A))
eigvals(A::Hermitian{T,<:BandedMatrix{T}}) where T<:Real = eigvals!(tridiagonalize(A))
eigvals(A::Hermitian{T,<:BandedMatrix{T}}) where T<:Complex = eigvals!(tridiagonalize(A))

eigvals!(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigvals!(tridiagonalize!(A))
function eigvals!(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T<:Real
    n = size(A, 1)
    @assert n == size(B, 1)
    @assert A.uplo == B.uplo
    # compute split-Cholesky factorization of B.
    kb = bandwidth(B)
    B_data = symbandeddata(B)
    pbstf!(B.uplo, n, kb, B_data)
    # convert to a regular symmetric eigenvalue problem.
    ka = bandwidth(A)
    A_data = symbandeddata(A)
    X = Array{T}(undef,0,0)
    work = Vector{T}(undef,2n)
    sbgst!('N', A.uplo, n, ka, kb, A_data, B_data, X, work)
    # compute eigenvalues of symmetric eigenvalue problem.
    eigvals!(A)
end

function eigvals(A::Symmetric{<:Any,<:BandedMatrix}, B::Symmetric{<:Any,<:BandedMatrix})
    AA = if bandwidth(A) >= bandwidth(B)
        copy(A)
    else
        copyto!(similar(B), A)
    end
    eigvals!(AA, copy(B))
end
