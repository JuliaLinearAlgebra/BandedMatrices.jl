


ql(A::BandedMatrix{T}) where T = ql!(BandedMatrix{float(T)}(A, (max(bandwidth(A,1),bandwidth(A,1)+bandwidth(A,2)+size(A,1)-size(A,2)),bandwidth(A,2))))
ql(A::Tridiagonal{T}) where T = ql!(BandedMatrix{float(T)}(A, (2,1)))

ql!(A::BandedMatrix) = banded_ql!(A)

function banded_ql!(L::BandedMatrix{T}) where T
    D = bandeddata(L)
    l,u = bandwidths(L)
    ν = l+u+1
    m,n=size(L)
    τ = zeros(T, min(m,n))

    for k = n:-1:max((n - m + 1 + (T<:Real)),1)
        μ = m+k-n
        x = view(D,u+1+μ-k:-1:max(1,u-k+2), k)
        τk = reflector!(x)
        τ[k-n+min(m,n)] = τk
        N = length(x)
        for j = max(1,μ-l):k-1
            reflectorApply!(x, τk, view(D, u+1+μ-j:-1:u+2+μ-j-N,j))
        end
    end
    QL(L, τ)
end

function lmul!(A::QLPackedQ{<:Any,<:BandedMatrix}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    begin
        for k = max(nA - mA + 1,1):nA
            μ = mA+k-nA
            for j = 1:nB
                vBj = B[μ,j]
                for i = max(1,k-u):μ-1
                    vBj += conj(D[i-k+u+1,k])*B[i,j]
                end
                vBj = A.τ[k-nA+min(mA,nA)]*vBj
                B[μ,j] -= vBj
                for i = max(1,k-u):μ-1
                    B[i,j] -= D[i-k+u+1,k]*vBj
                end
            end
        end
    end
    B
end


function lmul!(adjA::Adjoint{<:Any,<:QLPackedQ{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    @inbounds begin
        for k = nA:-1:max(nA - mA + 1,1)
            μ = mA+k-nA
            for j = 1:nB
                vBj = B[μ,j]
                for i = max(1,k-u):μ-1
                    vBj += conj(D[i-k+u+1,k])*B[i,j]
                end
                vBj = conj(A.τ[k-nA+min(mA,nA)])*vBj
                B[μ,j] -= vBj
                for i = max(1,k-u):μ-1
                    B[i,j] -= D[i-k+u+1,k]*vBj
                end
            end
        end
    end
    B
end

### QBc/QcBc
function rmul!(A::AbstractMatrix,Q::QLPackedQ{<:Any,<:BandedMatrix})
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    l,u = bandwidths(Qfactors)
    D = Qfactors.data
    @inbounds begin
        for k = nQ:-1:max(nQ - mQ + 1,1)
            μ = mQ+k-nQ
            for i = 1:mA
                vAi = A[i,μ]
                for j = max(1,k-u):μ-1
                    vAi += A[i,j]*D[j-k+u+1,k]
                end
                vAi = vAi*Q.τ[k-nQ+min(mQ,nQ)]
                A[i,μ] -= vAi
                for j = max(1,k-u):μ-1
                    A[i,j] -= vAi*conj(D[j-k+u+1,k])
                end
            end
        end
    end
    A
end

### AQc
function rmul!(A::AbstractMatrix, adjQ::Adjoint{<:Any,<:QLPackedQ{<:Any,<:BandedMatrix}})
    Q = adjQ.parent
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    l,u = bandwidths(Qfactors)
    D = Qfactors.data
    @inbounds begin
        for k = max(nQ - mQ + 1,1):nQ
            μ = mQ+k-nQ
            for i = 1:mA
                vAi = A[i,μ]
                for j = max(1,k-u):μ-1
                    vAi += A[i,j]*D[j-k+u+1,k]
                end
                vAi = vAi*conj(Q.τ[k-nQ+min(mQ,nQ)])
                A[i,μ] -= vAi
                for j = max(1,k-u):μ-1
                    A[i,j] -= vAi*conj(D[j-k+u+1,k])
                end
            end
        end
    end
    A
end




function _banded_widerect_ldiv!(A::QL, B)
    error("Not implemented")
end
function _banded_longrect_ldiv!(A::QL, B)
    error("Not implemented")
end
function _banded_square_ldiv!(A::QL, B)
    L = A.factors
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(LowerTriangular(L), B)
    B
end

for Typ in (:StridedVector, :StridedMatrix, :AbstractVecOrMat) 
    @eval function ldiv!(A::QL{T,<:BandedMatrix}, B::$Typ{T}) where T
        m, n = size(A)
        if m == n
            _banded_square_ldiv!(A, B)
        elseif n > m
            _banded_widerect_ldiv!(A, B)
        else
            _banded_longrect_ldiv!(A, B)
        end
    end
end
