## Banded LDLᵀ decomposition

@inline bandwidth(A::LDLt{T,Symmetric{T,M}}, k) where {T,M<:BandedMatrix{T}} = bandwidth(A.data, k)

factorize(S::Symmetric{<:Any,<:BandedMatrix}) = ldlt(S)

function ldlt!(F::Symmetric{T,M}) where {T,M<:BandedMatrix{T}}
    A = F.data
    n = size(A, 1)
    b = bandwidth(F)
    if F.uplo == 'L'
        @inbounds for j = 1:n
            @simd for k = max(1,j-b-1):j-1
                A[j,j] -= abs2(A[j,k])*A[k,k]
            end
            for i = j+1:min(n,j+b)
                @simd for k = max(1,j-b-1):j-1
                    A[i,j] -= A[i,k]*A[j,k]*A[k,k]
                end
                A[i,j] /= A[j,j]
            end
        end
    elseif F.uplo == 'U'
        @inbounds for j = 1:n
            @simd for k = max(1,j-b-1):j-1
                A[j,j] -= abs2(A[k,j])*A[k,k]
            end
            for i = j+1:min(n,j+b)
                @simd for k = max(1,j-b-1):j-1
                    A[j,i] -= A[k,i]*A[k,j]*A[k,k]
                end
                A[j,i] /= A[j,j]
            end
        end
    end
    return LDLt{T,Symmetric{T,M}}(F)
end

function ldlt(A::Symmetric{T,M}) where {T,M<:BandedMatrix{T}}
    S = typeof(zero(T)/one(T))
    uplo = A.uplo == 'U' ? :U : :L
    return S == T ? ldlt!(copy(A)) : ldlt!(Symmetric(BandedMatrix{S}(A), uplo))
end


function ldiv!(S::LDLt{T,Symmetric{T,M}}, B::AbstractVecOrMat{T}) where {T,M<:BandedMatrix{T}}
    require_one_based_indexing(B)
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
