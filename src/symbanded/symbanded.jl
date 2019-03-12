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




@inline inbands_getindex(A::Symmetric{<:Any, <:BandedMatrix}, k::Integer, j::Integer) =
    parent(A).data[bandwidth(A) - abs(k-j) + 1, max(k,j)]



symmetriclayout(layout::BandedColumns, uplo) = SymmetricLayout(layout,uplo)
symmetriclayout(layout::BandedRows, uplo) = SymmetricLayout(layout,uplo)

hermitianlayout(::Type{<:Complex}, layout::BandedColumns, uplo) = HermitianLayout(layout,uplo)
hermitianlayout(::Type{<:Real}, layout::BandedColumns, uplo) = SymmetricLayout(layout,uplo)
hermitianlayout(::Type{<:Complex}, layout::BandedRows, uplo) = HermitianLayout(layout,uplo)
hermitianlayout(::Type{<:Real}, layout::BandedRows, uplo) = SymmetricLayout(layout,uplo)


isbanded(A::HermOrSym) = isbanded(parent(A))

bandwidth(A::HermOrSym) = ifelse(A.uplo == 'U', bandwidth(parent(A),2), bandwidth(parent(A),1))
bandwidths(A::HermOrSym) = (bandwidth(A), bandwidth(A))

Base.replace_in_print_matrix(A::HermOrSym{<:Any,<:AbstractBandedMatrix}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A) ≤ j-i ≤ bandwidth(A) ? s : Base.replace_with_centered_mark(s)

function symbandeddata(A)
    M = MemoryLayout(A)
    B = symmetricdata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    if M.uplo == 'U'
        view(D, 1:u+1, :)
    else
        m = size(D,1)
        view(D, u+1:u+l+1, :)
    end
end

function hermbandeddata(A)
    M = MemoryLayout(A)
    B = hermitiandata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    if M.uplo == 'U'
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


function materialize!(M::BlasMatMulVec{<:SymmetricLayout{<:BandedColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat})
    S, α, A, x, β, y = M.style_A, M.α, M.A, M.B, M.β, M.C
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l = bandwidth(A)
    l ≥ 0 || return lmul!(β, y)
    _banded_sbmv!(S.uplo, α, A, x, β, y)
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

function materialize!(M::BlasMatMulVec{<:HermitianLayout{<:BandedColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat})
    S, α, A, x, β, y = M.style_A, M.α, M.A, M.B, M.β, M.C
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square"))
    (length(y) ≠ m || length(x) ≠ n) && throw(DimensionMismatch("*"))
    l = bandwidth(A)
    l ≥ 0 || return lmul!(β, y)
    _banded_hbmv!(S.uplo, α, A, x, β, y)
end


## eigvals routine


function _tridiagonalize!(A::AbstractMatrix{T}, S::SymmetricLayout{<:BandedColumnMajor}) where T
    n=size(A, 1)
    d = Vector{T}(undef,n)
    e = Vector{T}(undef,n-1)
    Q = Matrix{T}(undef,0,0)
    work = Vector{T}(undef,n)
    sbtrd!('N', S.uplo, size(A,1), bandwidth(A), symbandeddata(A), d, e, Q, work)
    SymTridiagonal(d,e)
end

tridiagonalize!(A::AbstractMatrix) = _tridiagonalize!(A, MemoryLayout(A))
tridiagonalize(A::AbstractMatrix) = tridiagonalize!(copy(A))

eigvals!(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigvals!(tridiagonalize!(A))
eigvals(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigvals!(copy(A))

function eigvals!(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T<:Real
    n = size(A, 1)
    @assert n == size(B, 1)
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
    # compute eigenvalues (and eigenvectors) of symmetric eigenvalue problem.
    eigvals!(A)
end

eigvals(A::Symmetric{<:Any,<:BandedMatrix}, B::Symmetric{<:Any,<:BandedMatrix}) = eigvals!(copy(A), copy(B))

function eigen!(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real
    n = size(A, 1)
    w = Vector{T}(undef, n)
    Z = Matrix{T}(undef, n, n)
    kd = bandwidth(A)
    work = Vector{T}(undef, max(1, 3*n-2))
    sbev!('V', A.uplo, n, kd, symbandeddata(A), w, Z, work)
    Eigen(w, Z)
end

eigen(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigen!(copy(A))

function eigen!(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T <: Real
    n = size(A, 1)
    @assert n == size(B, 1)
    @assert A.uplo == B.uplo
    w = Vector{T}(undef, n)
    Z = Matrix{T}(undef, n, n)
    ka = bandwidth(A)
    kb = bandwidth(B)
    work = Vector{T}(undef, 3*n)
    sbgv!('V', A.uplo, n, ka, kb, symbandeddata(A), symbandeddata(B), w, Z, work)
    GeneralizedEigen(w, Z)
end

eigen(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigen!(copy(A), copy(B))
