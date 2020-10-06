function _dot(x::AbstractVector, A::BandedMatrix, y::AbstractVector)
    require_one_based_indexing(x, y)
    (axes(x)..., axes(y)...) == axes(A) || throw(DimensionMismatch())
    l,u = bandwidths(A)
    M = size(A,1)
    T = typeof(dot(first(x), first(A), first(y)))
    s = zero(T)
    i₁ = first(eachindex(x))
    x₁ = first(x)
    @inbounds for j in eachindex(y)
        yj = y[j]
        if !iszero(yj)
            temp = zero(adjoint(A[i₁,j]) * x₁)
            @simd for i in max(1,j-u):min(M,j+l)
                temp += adjoint(A[i,j]) * x[i]
            end
            s += dot(temp, yj)
        end
    end
    s
end

dot(x::AbstractVector, A::BandedMatrix, y::AbstractVector) =
    _dot(x, A, y)

function dot(x::AbstractVector, A::BandedMatrix{<:Any,<:AbstractFill}, y::AbstractVector)
    require_one_based_indexing(x, y)
    l,u = bandwidths(A)
    l == -u || return _dot(x, A, y)

    (axes(x)..., axes(y)...) == axes(A) || throw(DimensionMismatch())
    T = typeof(dot(first(x), first(A), first(y)))
    Av = unique_value(A.data)
    iszero(Av) && return zero(T)

    M,N = size(A)
    @inbounds begin
        xv = view(x, max(1,l+1):min(M,N+l))
        yv = view(y, max(1,u+1):min(M+u,N))

        Av*dot(xv, yv)
    end
end
