## Banded LU decomposition

# This file is based on a part of Julia. License is MIT: https://julialang.org/license

####################
# Banded LU Factorization #
# This is like LU but factors does not actually
# permute the rows.  
####################
struct BandedLU{T,S<:AbstractMatrix{T}} <: Factorization{T}
    factors::S
    ipiv::Vector{BlasInt}
    info::BlasInt

    function BandedLU{T,S}(factors, ipiv, info) where {T,S<:AbstractMatrix{T}}
        require_one_based_indexing(factors)
        new{T,S}(factors, ipiv, info)
    end
end
function BandedLU(factors::AbstractMatrix{T}, ipiv::Vector{BlasInt}, info::BlasInt) where {T}
    BandedLU{T,typeof(factors)}(factors, ipiv, info)
end
function BandedLU{T}(factors::AbstractMatrix, ipiv::AbstractVector{<:Integer}, info::Integer) where {T}
    BandedLU(convert(AbstractMatrix{T}, factors),
       convert(Vector{BlasInt}, ipiv),
       BlasInt(info))
end

# iteration for destructuring into components
Base.iterate(S::BandedLU) = (S.L, Val(:U))
Base.iterate(S::BandedLU, ::Val{:U}) = (S.U, Val(:p))
Base.iterate(S::BandedLU, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::BandedLU, ::Val{:done}) = nothing

adjoint(F::BandedLU) = Adjoint(F)
transpose(F::BandedLU) = Transpose(F)

lu(S::BandedLU) = S

function BandedLU{T}(F::BandedLU) where T
    M = convert(AbstractMatrix{T}, F.factors)
    BandedLU{T,typeof(M)}(M, F.ipiv, F.info)
end
BandedLU{T,S}(F::BandedLU) where {T,S} = BandedLU{T,S}(convert(S, F.factors), F.ipiv, F.info)
Factorization{T}(F::BandedLU{T}) where {T} = F
Factorization{T}(F::BandedLU) where {T} = BandedLU{T}(F)

copy(A::BandedLU{T,S}) where {T,S} = BandedLU{T,S}(copy(A.factors), copy(A.ipiv), A.info)

size(A::BandedLU)    = size(getfield(A, :factors))
size(A::BandedLU, i) = size(getfield(A, :factors), i)

Base.propertynames(F::BandedLU, private::Bool=false) =
    (:L, :U, :p, :P, (private ? fieldnames(typeof(F)) : ())...)

LinearAlgebra.issuccess(F::BandedLU) = F.info == 0

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::BandedLU)
    if issuccess(F)
        summary(io, F); println(io)
        println(io, "L factor:")
        show(io, mime, F.L)
        println(io, "\nU factor:")
        show(io, mime, F.U)
    else
        print(io, "Failed factorization of type $(typeof(F))")
    end
end


/(B::AbstractMatrix, A::BandedLU) = copy(transpose(transpose(A) \ transpose(B)))

# Conversions
AbstractMatrix(F::BandedLU) = (F.L * F.U)[invperm(F.p),:]
AbstractArray(F::BandedLU) = AbstractMatrix(F)
Matrix(F::BandedLU) = Array(AbstractArray(F))
Array(F::BandedLU) = Matrix(F)

##
function lu!(A::BandedMatrix{T}, pivot::Union{Val{false}, Val{true}} = Val(true);
             check::Bool = true) where T<:BlasFloat
    if pivot === Val(false)
        return banded_lufact!(A, pivot; check = check)
    end
    m= size(A,1)
    l,u = bandwidths(A) # l of the bands are ignored and overwritten
    _, ipiv = LAPACK.gbtrf!(l, u-l, m, bandeddata(A))
    return BandedLU{T,typeof(A)}(A, ipiv, zero(BlasInt))
end

lu!(A::AbstractBandedMatrix, pivot::Union{Val{false}, Val{true}} = Val(true); check::Bool = true) =
    banded_lufact!(A, pivot; check = check)

function lu(A::Union{AbstractBandedMatrix{T}, AbstractBandedMatrix{Complex{T}}, Adjoint{T,<:AbstractBandedMatrix{T}}, Adjoint{Complex{T},<:AbstractBandedMatrix{Complex{T}}},
                    Transpose{T,<:AbstractBandedMatrix{T}}, Transpose{Complex{T},<:AbstractBandedMatrix{Complex{T}}}},
    pivot::Union{Val{false}, Val{true}} = Val(true);
    check::Bool = true) where {T<:Real}
    l,u = bandwidths(A)
    lu!(BandedMatrix{float(eltype(A))}(A,(l,l+u)), pivot; check = check)
end

# Jesus christ someone loves to write "StridedMatrix" for no reason in Base!
function getproperty(F::BandedLU{T}, d::Symbol) where T
    m, n = size(F)
    if d == :L
        # not clear how to get it from F.factors so we 
        # form it in the most insane way possible
        Ai = F\Matrix(I,n,m)
        tril!((F.P/Ai)/F.U)
    elseif d == :U
        return triu!(getfield(F, :factors)[1:min(m,n), 1:n])
    elseif d == :p
        return ipiv2perm(getfield(F, :ipiv), m)
    elseif d == :P
        return Matrix{T}(I, m, m)[:,invperm(F.p)]
    else
        getfield(F, d)
    end
end

# function banded_lufact!(A::AbstractMatrix{T}, ::Val{Pivot} = Val(true);
#                          check::Bool = true) where {T,Pivot}
#     m, n = size(A)
#     l, u = bandwidths(A)
#     minmn = min(m,n)
#     info = 0
#     ipiv = Vector{BlasInt}(undef, minmn)
#     @inbounds begin
#         for k = 1:minmn
#             # find index max
#             kp = k
#             if Pivot
#                 amax = abs(zero(T))
#                 for i = k:min(k+l,m)
#                     absi = abs(A[i,k])
#                     if absi > amax
#                         kp = i
#                         amax = absi
#                     end
#                 end
#             end
#             ipiv[k] = kp
#             if !iszero(A[kp,k])
#                 if k != kp
#                     # Interchange
#                     for i = rowrange(A,k)
#                         tmp = A[k,i]
#                         A[k,i] = A[kp,i]
#                         A[kp,i] = tmp
#                     end
#                 end
#                 # Scale first column
#                 Akkinv = inv(A[k,k])
#                 for i = k+1:min(k+l,m)
#                     A[i,k] *= Akkinv
#                 end
#             elseif info == 0
#                 info = k
#             end
#             # Update the rest
#             for j = k+1:min(k+u,n)
#                 for i = k+1:min(k+l,m)
#                     A[i,j] -= A[i,k]*A[k,j]
#                 end
#             end
#         end
#     end
#     check && checknonsingular(info)
#     return BandedLU{T,typeof(A)}(A, ipiv, convert(BlasInt, info))
# end


@lazyldiv BandedMatrix
# @lazyldiv BandedLU
