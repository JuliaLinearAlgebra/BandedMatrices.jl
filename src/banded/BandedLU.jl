## Banded LU decomposition


# LU factorisation with pivoting. This makes a copy!
# function lu(A::BandedMatrix{T}) where {T<:Number}
#     # copy to a blas type that allows calculation of the factorisation in place.
#     S = _promote_to_blas_type(T, T)
#     # copy into larger array of size (2l+u*1)×n, i.e. l additional rows
#     m, n = size(A)
#     data = Array{S}(undef, 2*A.l+A.u+1, n)
#     data[(A.l+1):end, :] = A.data
#     data, ipiv = LAPACK.gbtrf!(m, A.l, A.u, data)
#     BandedLU{S}(data, ipiv, A.l, A.u, m)
# end

##
function lu!(A::BandedMatrix{T}, pivot::Union{Val{false}, Val{true}} = Val(true);
             check::Bool = true) where T<:BlasFloat
    if pivot === Val(false)
        return banded_lufact!(A, pivot; check = check)
    end
    m= size(A,1)
    l,u = bandwidths(A) # l of the bands are ignored and overwritten
    _, ipiv = LAPACK.gbtrf!(l, u-l, m, bandeddata(A))
    return LU{T,typeof(A)}(A, ipiv, zero(BlasInt))
end

lu!(A::AbstractBandedMatrix, pivot::Union{Val{false}, Val{true}} = Val(true); check::Bool = true) =
    banded_lufact!(A, pivot; check = check)

function lu(A::Union{AbstractBandedMatrix{T}, AbstractBandedMatrix{Complex{T}}},
    pivot::Union{Val{false}, Val{true}} = Val(true);
    check::Bool = true) where {T<:AbstractFloat}
    l,u = bandwidths(A)
    lu!(BandedMatrix(A,(l,l+u)), pivot; check = check)
end

# Jesus christ someone loves to write "StridedMatrix" for no reason in Base!
function getproperty(F::LU{T,<:AbstractBandedMatrix}, d::Symbol) where T
    m, n = size(F)
    if d == :L
        L = tril!(getfield(F, :factors)[1:m, 1:min(m,n)])
        for i = 1:min(m,n); L[i,i] = one(T); end
        return L
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

function banded_lufact!(A::AbstractMatrix{T}, ::Val{Pivot} = Val(true);
                         check::Bool = true) where {T,Pivot}
    m, n = size(A)
    l, u = bandwidths(A)
    minmn = min(m,n)
    info = 0
    ipiv = Vector{BlasInt}(undef, minmn)
    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if Pivot
                amax = abs(zero(T))
                for i = k:min(k+l,m)
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            ipiv[k] = kp
            if !iszero(A[kp,k])
                if k != kp
                    # Interchange
                    for i = rowrange(A,k)
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:min(k+l,m)
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end
            # Update the rest
            for j = k+1:min(k+u,n)
                for i = k+1:min(k+l,m)
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end
    end
    check && checknonsingular(info)
    return LU{T,typeof(A)}(A, ipiv, convert(BlasInt, info))
end


@lazyldiv BandedMatrix
# @lazyldiv BandedLU
