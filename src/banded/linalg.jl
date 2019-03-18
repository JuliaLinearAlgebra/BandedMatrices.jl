## Method definitions for generic eltypes - will make copies

# Direct and transposed algorithms

function _copyto!(_, dest::AbstractVecOrMat, L::ArrayLdivArray{<:BandedColumnMajor})
    A, B = L.args
    checksquare(A)
    dest ≡ B || copyto!(dest, B)
    ldiv!(factorize(A), dest)
end

function _copyto!(_, dest::AbstractVecOrMat, L::ArrayLdivArray{<:BandedRowMajor})
    A, B = L.args
    copyto!(dest, Mul(transpose(factorize(transpose(A))), B))
end

function _copyto!(_, dest::AbstractVecOrMat, L::ArrayLdivArray{<:ConjLayout{<:BandedRowMajor}})
    A, B = L.args
    copyto!(dest, Mul(factorize(A')', B))
end



# Direct and transposed algorithms
function ldiv!(A::LU{T,<:BandedMatrix}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('N', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(A::LU{<:Any,<:BandedMatrix}, B::AbstractVecOrMat)
    _apply_ipiv!(A, B)
    ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), B))
end

function ldiv!(transA::Transpose{T,<:LU{T,<:BandedMatrix}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
    A = transA.parent
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('T', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(transA::Transpose{<:Any,<:LU{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    A = transA.parent
    ldiv!(transpose(UnitLowerTriangular(A.factors)), ldiv!(transpose(UpperTriangular(A.factors)), B))
    _apply_inverse_ipiv!(A, B)
end

ldiv!(adjF::Adjoint{T,<:LU{T,<:BandedMatrix}}, B::AbstractVecOrMat{T}) where {T<:Real} =
    (F = adjF.parent; ldiv!(transpose(F), B))
function ldiv!(adjA::Adjoint{T,<:LU{T,<:BandedMatrix}}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
    A = adjA.parent
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('C', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(adjA::Adjoint{<:Any,<:LU{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    A = adjA.parent
    ldiv!(adjoint(UnitLowerTriangular(A.factors)), ldiv!(adjoint(UpperTriangular(A.factors)), B))
    _apply_inverse_ipiv!(A, B)
end


factorize(A::BandedMatrix) = lu(A)
