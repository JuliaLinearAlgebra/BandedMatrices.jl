## Banded LDLᵀ decomposition

@inline bandwidth(A::LDLt{T,Symmetric{T,M}}, k) where {T,M<:BandedMatrix{T}} = bandwidth(A.data, k)

factorize(S::Symmetric{<:Any,<:BandedMatrix}) = ldlt(S)

function ldlt(A::Symmetric{T,M}) where {T<:Real,M<:BandedMatrix{T}}
    F = Symmetric(similar(A.data), :L)
    B = F.data
    n = size(A, 1)
    b = bandwidth(A, 1)
    for j = 1:n
        B[j,j] = A[j,j]
        for k = max(1,j-b-1):j-1
            B[j,j] -= B[j,k]^2*B[k,k]
        end
        for i = j+1:min(n,j+b)
            B[i,j] = A[i,j]
            for k = max(1,j-b-1):j-1
                B[i,j] -= B[i,k]*B[j,k]*B[k,k]
            end
            B[i,j] /= B[j,j]
        end
    end
    return LDLt{T,Symmetric{T,M}}(F)
end

function ldiv!(S::LDLt{T,Symmetric{T,M}}, B::AbstractVecOrMat{T}) where {T,M<:BandedMatrix{T}}
    @assert !has_offset_axes(B)
    n, nrhs = size(B, 1), size(B, 2)
    if size(S, 1) != n
        throw(DimensionMismatch("Matrix has dimensions $(size(S)) but right hand side has first dimension $n"))
    end
    A = S.data
    b = bandwidth(A, 1)
    @inbounds for l = 1:nrhs
        for j = 1:n
            @simd for k = max(1,j-b-1):j-1
                B[j,l] -= A[j,k]*B[k,l]
            end
        end
        @simd for j = 1:n
            B[j,l] /= A[j,j]
        end
        for j = n:-1:1
            @simd for k = j+1:min(n,j+b)
                B[j,l] -= A[k,j]*B[k,l]
            end
        end
    end
    B
end

function logabsdet(F::LDLt{T,Symmetric{T,M}}) where {T,M<:BandedMatrix{T}}
    it = (F.data[i,i] for i in 1:size(F, 1))
    return sum(log∘abs, it), prod(sign, it)
end
