AdjointQType = isdefined(LinearAlgebra, :AdjointQ) ? LinearAlgebra.AdjointQ : Adjoint

_qr(::AbstractBandedLayout, ax, A) = _banded_qr(ax, A)
_banded_qr(_, A) = qr!(BandedMatrix{float(eltype(A))}(A, (bandwidth(A,1),bandwidth(A,1)+bandwidth(A,2))))

qr(A::Tridiagonal{T}) where T = qr!(BandedMatrix{float(T)}(A, (1,2)))
_qr!(::AbstractBandedLayout, _, A) = banded_qr!(A)


function _banded_qr!(R::AbstractMatrix{T}, τ) where T
    m,n=size(R)
    _banded_qr!(R, τ, min(m - 1 + !(T<:Real), n))
end

function _banded_qr!(R::AbstractMatrix, τ, ncols)
    D = bandeddata(R)
    l,u = bandwidths(R)
    ν = l+u+1
    m,n=size(R)

    for k = 1:ncols
        x = view(D,u+1:min(ν,m-k+u+1), k)
        τk = reflector!(x)
        τ[k] = τk
        for j = 1:min(u,n-k)
            reflectorApply!(x, τk, view(D, u+1-j:min(ν-j,m-k-j+u+1), k+j:k+j))
        end
    end
    R, τ
end

function banded_qr!(R::AbstractMatrix{T}, τ) where T
   _banded_qr!(R, τ)
   QR(R, convert(Vector{T},τ)) # TODO: remove convert
end

banded_qr!(R::AbstractMatrix{T}) where T = banded_qr!(R, zeros(T, min(size(R)...)))


function banded_qr_lmul!(A, B)
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = bandeddata(Afactors)
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

function banded_qr_lmul!(adjA::AdjointQType, B)
    require_one_based_indexing(B)
    A = parent(adjA)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = bandeddata(Afactors)
    @inbounds begin
        for j = 1:nB
            cs = colsupport(B,j)
            for k = max(1,minimum(cs)):min(mA,nA,maximum(cs)+l,length(A.τ))
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

function banded_qr_rmul!(A, Q)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    l,u = bandwidths(Qfactors)
    D = bandeddata(Qfactors)
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
function banded_qr_rmul!(A, adjQ::AdjointQType)
    Q = parent(adjQ)
    mQ, nQ = size(Q.factors)
    mA, nA = size(A,1), size(A,2)
    if nA != mQ
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but matrix Q has dimensions ($mQ, $nQ)"))
    end
    Qfactors = Q.factors
    l,u = bandwidths(Qfactors)
    D = bandeddata(Qfactors)
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

banded_lmul!(A::QRPackedQ, B::AbstractVecOrMat) = banded_qr_lmul!(A, B)
banded_lmul!(adjA::AdjointQType{<:Any,<:QRPackedQ}, B::AbstractVecOrMat) = banded_qr_lmul!(adjA, B)
banded_rmul!(A::AbstractMatrix, Q::QRPackedQ) = banded_qr_rmul!(A, Q)
banded_rmul!(A::AbstractMatrix, adjQ::AdjointQType{<:Any,<:QRPackedQ}) = banded_qr_rmul!(A, adjQ)

lmul!(A::QRPackedQ{<:Any,<:AbstractBandedMatrix}, B::AbstractVector) = banded_lmul!(A,B)
lmul!(A::QRPackedQ{<:Any,<:AbstractBandedMatrix}, B::AbstractMatrix) = banded_lmul!(A,B)
lmul!(adjA::AdjointQType{<:Any,<:QRPackedQ{<:Any,<:AbstractBandedMatrix}}, B::AbstractVector) = banded_lmul!(adjA,B)
lmul!(adjA::AdjointQType{<:Any,<:QRPackedQ{<:Any,<:AbstractBandedMatrix}}, B::AbstractMatrix) = banded_lmul!(adjA,B)
lmul!(A::QRPackedQ{<:Any,BandedSubBandedMatrix{T,C,R,I1,I2}}, B::AbstractVector) where {T,C,R,I1<:AbstractUnitRange,I2<:AbstractUnitRange} =
    banded_lmul!(A,B)
lmul!(A::QRPackedQ{<:Any,BandedSubBandedMatrix{T,C,R,I1,I2}}, B::AbstractMatrix) where {T,C,R,I1<:AbstractUnitRange,I2<:AbstractUnitRange} =
    banded_lmul!(A,B)
lmul!(adjA::AdjointQType{T,<:QRPackedQ{T,<:BandedSubBandedMatrix{T,C,R,I1,I2,t}}}, B::AbstractVector) where {T,C,R,I1<:AbstractUnitRange,I2<:AbstractUnitRange,t} =
    banded_lmul!(adjA,B)
lmul!(adjA::AdjointQType{T,<:QRPackedQ{T,<:BandedSubBandedMatrix{T,C,R,I1,I2,t}}}, B::AbstractMatrix) where {T,C,R,I1<:AbstractUnitRange,I2<:AbstractUnitRange,t} =
    banded_lmul!(adjA,B)
# rmul!(A::AbstractMatrix, adjQ::AdjointQType{<:Any,<:QRPackedQ{<:Any,<:AbstractBandedMatrix}}) = banded_rmul!(A, adjA)
# rmul!(A::StridedMatrix, adjQ::AdjointQType{<:Any,<:QRPackedQ{<:Any,<:AbstractBandedMatrix}}) = banded_rmul!(A, adjA)
rmul!(A::StridedVecOrMat{T}, Q::QRPackedQ{T,B}) where {T<:BlasFloat,B<:AbstractBandedMatrix{T}} = banded_rmul!(A, Q)
rmul!(A::StridedVecOrMat{T}, adjQ::AdjointQType{<:Any,<:QRPackedQ{T,B}}) where {T<:BlasComplex,B<:AbstractBandedMatrix{T}} = banded_rmul!(A, adjQ)
rmul!(A::StridedVecOrMat{T}, adjQ::AdjointQType{<:Any,<:QRPackedQ{T,B}}) where {T<:BlasReal,B<:AbstractBandedMatrix{T}} = banded_rmul!(A, adjQ)


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

const UpperLayoutMatrix{T} = UpperTriangular{T,<:LayoutMatrix{T}}

for  BTyp in (:AbstractBandedMatrix, :BandedSubBandedMatrix), Typ in (:StridedVector, :StridedMatrix, :AbstractVecOrMat, :UpperLayoutMatrix, :LayoutMatrix)
    @eval function ldiv!(A::QR{T,<:$BTyp}, B::$Typ{T}) where T
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



materialize!(M::Lmul{<:QRPackedQLayout{<:AbstractBandedLayout}}) = banded_qr_lmul!(M.A,M.B)
materialize!(M::Lmul{<:AdjQRPackedQLayout{<:AbstractBandedLayout}}) = banded_qr_lmul!(M.A,M.B)

materialize!(M::Rmul{<:Any,<:QRPackedQLayout{<:AbstractBandedLayout}}) = banded_qr_rmul!(M.A,M.B)
materialize!(M::Rmul{<:Any,<:AdjQRPackedQLayout{<:AbstractBandedLayout}}) = banded_qr_rmul!(M.A,M.B)
