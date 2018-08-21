sbmv!(α::T,A::SymBandedMatrix{T},x::AbstractVector{T},β::T,y::AbstractVector{T}) where {T<:BlasFloat} =
  sbmv!('U',A.k,α,A.data,x,β,y)



function symbanded_A_mul_B!(c::AbstractVector{T},A::AbstractMatrix{T},b::AbstractVector{T}) where {T<:BlasFloat}
    n = size(A,1)

    @boundscheck if length(c) ≠ n || length(b) ≠ n
        throw(DimensionMismatch())
    end

    k = bandwidth(A,2)
    sbmv!('U',n,k,one(T),
            pointer(A),leadingdimension(A),pointer(b),stride(b,1),zero(T),pointer(c),stride(c,1))
    c
end


A_mul_B!(c::AbstractVector,A::SymBandedMatrix{T},b::AbstractVector) where {T} =
    symbanded_A_mul_B!(c,A,b)

## TODO

# - A_mul_B where A and B are banded matrices
# - Ac_mul_B!
