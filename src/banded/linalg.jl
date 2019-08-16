## Method definitions for generic eltypes - will make copies

# Direct and transposed algorithms

function materialize!(L::Ldiv{<:BandedColumnMajor})
    A, B = L.A, L.B
    checksquare(A)
    ldiv!(factorize(A), dest)
end

function copyto!(dest::AbstractVecOrMat, L::Ldiv{<:BandedRowMajor})
    A, B = L.A, L.B
    copyto!(dest, Mul(transpose(factorize(transpose(A))), B))
end

function copyto!(dest::AbstractVecOrMat, L::Ldiv{<:ConjLayout{<:BandedRowMajor}})
    A, B = L.A, L.B
    copyto!(dest, Mul(factorize(A')', B))
end



# Direct and transposed algorithms
function ldiv!(A::BandedLU{T,<:BandedMatrix}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('N', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(A::BandedLU, B::AbstractVecOrMat)
    _apply_ipiv!(A, B)
    ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), B))
end

function ldiv!(A::BandedLU{T}, B::AbstractVecOrMat{Complex{T}}) where T<:Real
    # c2r = reshape(transpose(reinterpret(T, reshape(B, (1, length(B))))), size(B, 1), 2*size(B, 2))
    a = real.(B)
    b = imag.(B)
    ldiv!(A, a)
    ldiv!(A, b)
    B .= a .+ im.*b
end

function ldiv!(transA::Transpose{T,<:BandedLU{T,<:BandedMatrix}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
    A = transA.parent
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('T', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(transA::Transpose{<:Any,<:BandedLU{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    A = transA.parent
    ldiv!(transpose(UnitLowerTriangular(A.factors)), ldiv!(transpose(UpperTriangular(A.factors)), B))
    _apply_inverse_ipiv!(A, B)
end

ldiv!(adjF::Adjoint{T,<:BandedLU{T,<:BandedMatrix}}, B::AbstractVecOrMat{T}) where {T<:Real} =
    (F = adjF.parent; ldiv!(transpose(F), B))
function ldiv!(adjA::Adjoint{T,<:BandedLU{T,<:BandedMatrix}}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
    A = adjA.parent
    m = size(A.factors,1)
    l,u = bandwidths(A.factors)
    data = bandeddata(A.factors)
    LAPACK.gbtrs!('C', l, u-l, m, data, A.ipiv, B)
end

function ldiv!(adjA::Adjoint{<:Any,<:BandedLU{<:Any,<:BandedMatrix}}, B::AbstractVecOrMat)
    error("Implement")
    A = adjA.parent
    ldiv!(adjoint(UnitLowerTriangular(A.factors)), ldiv!(adjoint(UpperTriangular(A.factors)), B))
    _apply_inverse_ipiv!(A, B)
end

\(A::Adjoint{<:Any,<:BandedLU}, B::Adjoint{<:Any,<:AbstractVecOrMat}) = A \ copy(B)
\(A::Transpose{<:Any,<:BandedLU}, B::Transpose{<:Any,<:AbstractVecOrMat}) = A \ copy(B)

factorize(A::BandedMatrix) = lu(A)
