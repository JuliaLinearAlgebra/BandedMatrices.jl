
qr(A::BandedMatrix) = banded_qr!(BandedMatrix(A, (bandwidth(A,1),bandwidth(A,1)+bandwidth(A,2))))
qr(A::Tridiagonal) = banded_qr!(BandedMatrix(A, (1,2)))

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

function lmul!(adjA::Adjoint{<:Any,<:QRPackedQ{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    @assert !has_offset_axes(B)
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
