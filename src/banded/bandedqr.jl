
qr(A::BandedMatrix{T}) where T = qr!(BandedMatrix{float(T)}(A, (bandwidth(A,1),bandwidth(A,1)+bandwidth(A,2))))
qr(A::Tridiagonal{T}) where T = qr!(BandedMatrix{float(T)}(A, (1,2)))

qr!(A::BandedMatrix) = banded_qr!(A)


function banded_qr!(R::BandedMatrix{T}) where T
    D = bandeddata(R)
    l,u = bandwidths(R)
    ν = l+u+1
    m,n=size(R)
    τ = zeros(T, min(m,n))

    for k = 1:min(m - 1 + !(T<:Real), n)
        x = view(D,u+1:min(ν,m-k+u+1), k)
        τk = reflector!(x)
        τ[k] = τk
        for j = 1:min(u,n-k)
            reflectorApply!(x, τk, view(D, u+1-j:min(ν-j,m-k-j+u+1), k+j:k+j))
        end
    end
    QR(R, τ)
end

function lmul!(A::QRPackedQ{<:Any,<:BandedMatrix}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    @inbounds begin
        for k = min(mA,nA):-1:1
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:min(k+l,mB)
                    vBj += conj(D[i-k+u+1,k])*B[i,j]
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = k+1:min(k+l,mB)
                    B[i,j] -= D[i-k+u+1,k]*vBj
                end
            end
        end
    end
    B
end

function lmul!(adjA::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
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
        for k = 1:min(mA,nA)
            for j = 1:nB
                vBj = B[k,j]
                for i = k+1:min(k+l,mB)
                    vBj += conj(D[i-k+u+1,k])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = k+1:min(k+l,mB)
                    B[i,j] -= D[i-k+u+1,k]*vBj
                end
            end
        end
    end
    B
end

function banded_rmul!(A::AbstractMatrix,Q::QRPackedQ{<:Any,<:BandedMatrix})
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    l,u = bandwidths(Qfactors)
    D = Qfactors.data
    @inbounds begin
        for k = 1:min(mQ,nQ)
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:min(k+l,mQ)
                    vAi += A[i,j]*D[j-k+u+1,k]
                end
                vAi = vAi*Q.τ[k]
                A[i,k] -= vAi
                for j = k+1:min(k+l,nA)
                    A[i,j] -= vAi*conj(D[j-k+u+1,k])
                end
            end
        end
    end
    A
end

function banded_rmul!(A::AbstractMatrix, adjQ::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BandedMatrix}})
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
        for k = min(mQ,nQ):-1:1
            for i = 1:mA
                vAi = A[i,k]
                for j = k+1:min(k+l,mQ)
                    vAi += A[i,j]*D[j-k+u+1,k]
                end
                vAi = vAi*conj(Q.τ[k])
                A[i,k] -= vAi
                for j = k+1:min(k+l,nA)
                    A[i,j] -= vAi*conj(D[j-k+u+1,k])
                end
            end
        end
    end
    A
end

# rmul!(A::AbstractMatrix, adjQ::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BandedMatrix}}) = banded_rmul!(A, adjA)
# rmul!(A::StridedMatrix, adjQ::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BandedMatrix}}) = banded_rmul!(A, adjA)
rmul!(A::StridedVecOrMat{T}, Q::QRPackedQ{T,B}) where {T<:BlasFloat,B<:BandedMatrix{T}} = banded_rmul!(A, Q)
rmul!(A::StridedVecOrMat{T}, adjQ::Adjoint{<:Any,QRPackedQ{T,B}}) where {T<:BlasComplex,B<:BandedMatrix{T}} = banded_rmul!(A, adjQ)
rmul!(A::StridedVecOrMat{T}, adjQ::Adjoint{<:Any,QRPackedQ{T,B}}) where {T<:BlasReal,B<:BandedMatrix{T}} = banded_rmul!(A, adjQ)


function _banded_widerect_ldiv!(A::QR{T}, B) where T
    error("Not implemented")
end
function _banded_longrect_ldiv!(A::QR, B)
    m, n = size(A)
    R = A.factors
    lmul!(adjoint(A.Q), B)
    B̃ = view(B, 1:n, :)
    B̃ .= Ldiv(UpperTriangular(view(R, 1:n, 1:n)), B̃)
    B
end
function _banded_square_ldiv!(A::QR, B)
    R = A.factors
    lmul!(adjoint(A.Q), B)
    B .= Ldiv(UpperTriangular(R), B)
    B
end

for Typ in (:StridedVector, :StridedMatrix, :AbstractVecOrMat) 
    @eval function ldiv!(A::QR{T,<:BandedMatrix}, B::$Typ{T}) where T
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
