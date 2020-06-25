## Hermitian tridiagonal matrices
struct HermTridiagonal{S, T, U <: AbstractVector{S}, V <: AbstractVector{T}} <: AbstractMatrix{T}
    dv::U                        # diagonal
    ev::V                        # superdiagonal
    function HermTridiagonal{S, T, U, V}(dv, ev) where {S, T, U <: AbstractVector{S}, V <: AbstractVector{T}}
        require_one_based_indexing(dv, ev)
        if !(length(dv) - 1 <= length(ev) <= length(dv))
            throw(DimensionMismatch("superdiagonal has wrong length. Has length $(length(ev)), but should be either $(length(dv) - 1) or $(length(dv))."))
        end
        new{S,T,U,V}(dv,ev)
    end
end

HermTridiagonal(dv::U, ev::V) where {S,T,U<:AbstractVector{S},V<:AbstractVector{T}} = HermTridiagonal{S,T}(dv, ev)
HermTridiagonal{S,T}(dv::U, ev::V) where {S,T,U<:AbstractVector{S},V<:AbstractVector{T}} = HermTridiagonal{S,T,U,V}(dv, ev)
function HermTridiagonal{S,T}(dv::AbstractVector, ev::AbstractVector) where {S,T}
    HermTridiagonal(convert(AbstractVector{S}, dv)::AbstractVector{S},
                   convert(AbstractVector{T}, ev)::AbstractVector{T})
end

function HermTridiagonal(A::AbstractMatrix)
    if diag(A,1) == conj(diag(A,-1))
        HermTridiagonal(diag(A,0), diag(A,1))
    else
        throw(ArgumentError("matrix is not Hermitian; cannot convert to HermTridiagonal"))
    end
end

HermTridiagonal{S,T,U,V}(H::HermTridiagonal{S,T,U,V}) where {S,T,U<:AbstractVector{S},V<:AbstractVector{T}} = H
HermTridiagonal{S,T,U,V}(H::HermTridiagonal) where {S,T,U<:AbstractVector{S},V<:AbstractVector{T}} =
    HermTridiagonal(convert(U, H.dv)::U, convert(V, H.ev)::V)
HermTridiagonal{S,T}(H::HermTridiagonal{S,T}) where {S,T} = H
HermTridiagonal{S,T}(H::HermTridiagonal) where {S,T} =
    HermTridiagonal(convert(AbstractVector{S}, H.dv)::AbstractVector{S},
                   convert(AbstractVector{T}, H.ev)::AbstractVector{T})
HermTridiagonal(H::HermTridiagonal) = H

AbstractMatrix{T}(H::HermTridiagonal) where {T} =
    HermTridiagonal(convert(AbstractVector{T}, H.dv)::AbstractVector{T},
                   convert(AbstractVector{T}, H.ev)::AbstractVector{T})
function Matrix{T}(M::HermTridiagonal) where T
    n = size(M, 1)
    Mf = zeros(T, n, n)
    @inbounds begin
        @simd for i = 1:n-1
            Mf[i,i] = M.dv[i]
            Mf[i+1,i] = conj(M.ev[i])
            Mf[i,i+1] = M.ev[i]
        end
        Mf[n,n] = M.dv[n]
    end
    return Mf
end
Matrix(M::HermTridiagonal{S,T}) where {S,T} = Matrix{T}(M)
Array(M::HermTridiagonal) = Matrix(M)

size(A::HermTridiagonal) = (length(A.dv), length(A.dv))
function size(A::HermTridiagonal, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d<=2
        return length(A.dv)
    else
        return 1
    end
end

# For S<:HermTridiagonal, similar(S[, neweltype]) should yield a HermTridiagonal matrix.
# On the other hand, similar(S, [neweltype,] shape...) should yield a sparse matrix.
# The first method below effects the former, and the second the latter.
#similar(S::HermTridiagonal, ::Type{T}) where {T} = HermTridiagonal(similar(S.dv, T), similar(S.ev, T))
# The method below is moved to SparseArrays for now
# similar(S::HermTridiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)

#Elementary operations
for func in (:conj, :copy, :real, :imag)
    @eval ($func)(M::HermTridiagonal) = HermTridiagonal(($func)(M.dv), ($func)(M.ev))
end

transpose(H::HermTridiagonal) = Transpose(H)
adjoint(H::HermTridiagonal) = H
Base.copy(S::Adjoint{<:Any,<:HermTridiagonal}) = HermTridiagonal(map(x -> copy.(adjoint.(x)), (S.parent.dv, S.parent.ev))...)
Base.copy(S::Transpose{<:Any,<:HermTridiagonal}) = HermTridiagonal(map(x -> copy.(transpose.(x)), (S.parent.dv, S.parent.ev))...)

#=
function diag(M::HermTridiagonal, n::Integer=0)
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of n
    absn = abs(n)
    if absn == 0
        return copyto!(similar(M.dv, length(M.dv)), M.dv)
    elseif absn==1
        return copyto!(similar(M.ev, length(M.ev)), M.ev)
    elseif absn <= size(M,1)
        return fill!(similar(M.dv, size(M,1)-absn), 0)
    else
        throw(ArgumentError(string("requested diagonal, $n, must be at least $(-size(M, 1)) ",
            "and at most $(size(M, 2)) for an $(size(M, 1))-by-$(size(M, 2)) matrix")))
    end
end
=#

+(A::HermTridiagonal, B::HermTridiagonal) = HermTridiagonal(A.dv+B.dv, A.ev+B.ev)
-(A::HermTridiagonal, B::HermTridiagonal) = HermTridiagonal(A.dv-B.dv, A.ev-B.ev)
*(A::HermTridiagonal, B::Number) = HermTridiagonal(A.dv*B, A.ev*B)
*(B::Number, A::HermTridiagonal) = A*B
/(A::HermTridiagonal, B::Number) = HermTridiagonal(A.dv/B, A.ev/B)
==(A::HermTridiagonal, B::HermTridiagonal) = (A.dv==B.dv) && (A.ev==B.ev)

#=
@inline mul!(A::StridedVecOrMat, B::HermTridiagonal, C::StridedVecOrMat,
             alpha::Number, beta::Number) =
    _mul!(A, B, C, MulAddMul(alpha, beta))

@inline function _mul!(C::StridedVecOrMat, S::HermTridiagonal, B::StridedVecOrMat,
                          _add::MulAddMul)
    m, n = size(B, 1), size(B, 2)
    if !(m == size(S, 1) == size(C, 1))
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    if n != size(C, 2)
        throw(DimensionMismatch("second dimension of B, $n, doesn't match second dimension of C, $(size(C,2))"))
    end

    if m == 0
        return C
    elseif iszero(_add.alpha)
        return _rmul_or_fill!(C, _add.beta)
    end

    α = S.dv
    β = S.ev
    @inbounds begin
        for j = 1:n
            x₊ = B[1, j]
            x₀ = zero(x₊)
            # If m == 1 then β[1] is out of bounds
            β₀ = m > 1 ? zero(β[1]) : zero(eltype(β))
            for i = 1:m - 1
                x₋, x₀, x₊ = x₀, x₊, B[i + 1, j]
                β₋, β₀ = β₀, β[i]
                _modify!(_add, β₋*x₋ + α[i]*x₀ + β₀*x₊, C, (i, j))
            end
            _modify!(_add, β₀*x₀ + α[m]*x₊, C, (m, j))
        end
    end

    return C
end

(\)(T::HermTridiagonal, B::StridedVecOrMat) = ldlt(T)\B

# division with optional shift for use in shifted-Hessenberg solvers (hessenberg.jl):
ldiv!(A::HermTridiagonal, B::AbstractVecOrMat; shift::Number=false) = ldiv!(ldlt(A, shift=shift), B)
rdiv!(B::AbstractVecOrMat, A::HermTridiagonal; shift::Number=false) = rdiv!(B, ldlt(A, shift=shift))
=#


function eigen!(A::HermTridiagonal{S,T}) where {S,T}
    n = size(A, 1)
    D = ones(T, n)
    d = copy(A.dv)
    e = zeros(S, n-1)
    for i in 1:n-1
        e[i] = abs(A.ev[i])
        if e[i] != zero(S)
            D[i+1] = A.ev[i]/e[i]
        end
        if i < n-1
            A.ev[i+1] = A.ev[i+1]*D[i+1]
        end
    end
    Λ, V = eigen(SymTridiagonal(d, e))
    return Eigen(Λ, Diagonal(conj(D))*V)
end
eigen(A::HermTridiagonal{S,T}) where {S,T} = eigen!(copy(A))

eigvals!(A::HermTridiagonal{S,T}, kwargs...) where {S,T} = eigvals!(SymTridiagonal(A.dv, map(abs, A.ev)), kwargs...)
eigvals(A::HermTridiagonal{S,T}, kwargs...) where {S,T} = eigvals!(copy(A), kwargs...)

#Computes largest and smallest eigenvalue
eigmax(A::HermTridiagonal) = eigvals(A, size(A, 1):size(A, 1))[1]
eigmin(A::HermTridiagonal) = eigvals(A, 1:1)[1]

#Compute selected eigenvectors only corresponding to particular eigenvalues
eigvecs(A::HermTridiagonal) = eigen(A).vectors

function svdvals!(A::HermTridiagonal)
    vals = eigvals!(A)
    return sort!(map!(abs, vals, vals); rev=true)
end

#tril and triu

istriu(M::HermTridiagonal) = iszero(M.ev)
istril(M::HermTridiagonal) = iszero(M.ev)
iszero(M::HermTridiagonal) = iszero(M.ev) && iszero(M.dv)
isone(M::HermTridiagonal) = iszero(M.ev) && all(isone, M.dv)
isdiag(M::HermTridiagonal) = iszero(M.ev)

function tril!(M::HermTridiagonal, k::Integer=0)
    n = length(M.dv)
    if !(-n - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n - 1) and at most $(n - 1) in an $n-by-$n matrix")))
    elseif k < -1
        fill!(M.ev,0)
        fill!(M.dv,0)
        return Tridiagonal(conj(M.ev),M.dv,copy(M.ev))
    elseif k == -1
        fill!(M.dv,0)
        return Tridiagonal(conj(M.ev),M.dv,zero(M.ev))
    elseif k == 0
        return Tridiagonal(conj(M.ev),M.dv,zero(M.ev))
    elseif k >= 1
        return Tridiagonal(conj(M.ev),M.dv,copy(M.ev))
    end
end

function triu!(M::HermTridiagonal, k::Integer=0)
    n = length(M.dv)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n + 1) and at most $(n + 1) in an $n-by-$n matrix")))
    elseif k > 1
        fill!(M.ev,0)
        fill!(M.dv,0)
        return Tridiagonal(conj(M.ev),M.dv,copy(M.ev))
    elseif k == 1
        fill!(M.dv,0)
        return Tridiagonal(zero(M.ev),M.dv,M.ev)
    elseif k == 0
        return Tridiagonal(zero(M.ev),M.dv,M.ev)
    elseif k <= -1
        return Tridiagonal(conj(M.ev),M.dv,copy(M.ev))
    end
end

###################
# Generic methods #
###################

## structured matrix methods ##
function Base.replace_in_print_matrix(A::HermTridiagonal, i::Integer, j::Integer, s::AbstractString)
    i==j-1||i==j||i==j+1 ? s : Base.replace_with_centered_mark(s)
end

#logabsdet(A::HermTridiagonal; shift::Number=false) = logabsdet(ldlt(A; shift=shift))

function getindex(A::HermTridiagonal{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= size(A,2) && 1 <= j <= size(A,2))
        throw(BoundsError(A, (i,j)))
    end
    if i == j
        return A.dv[i]
    elseif i == j + 1
        return conj(A.ev[j])
    elseif i + 1 == j
        return A.ev[i]
    else
        return zero(T)
    end
end

function setindex!(A::HermTridiagonal, x, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    if i == j
        @inbounds A.dv[i] = x
    else
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j)"))
    end
    return x
end
