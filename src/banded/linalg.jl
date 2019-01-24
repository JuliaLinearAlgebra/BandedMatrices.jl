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
# function _copyto!(::AbstractStridedLayout, dest::AbstractVecOrMat{T}, M::MatMulVec{BandedLULayout}A::BandedLU{T}, B::AbstractVecOrMat{T}) where {T<:BlasFloat}
#     checksquare(A)
#     copyto!(dest, B)
#     gbtrs!('N', A.m, A.l, A.u, A.data, A.ipiv, dest)
# end

function ldiv!(A::BandedLU{T}, B::AbstractVecOrMat{S}) where {T<:Number, S<:Number}
     checksquare(A)
     AA, BB = _convert_to_blas_type(A, B)
     ldiv!(lu(AA), BB) # call BlasFloat versions
 end

function ldiv!(At::Transpose{T,BandedLU{T}}, B::AbstractVecOrMat{S}) where {T<:Number, S<:Number}
    A = parent(At)
    checksquare(A)
    AA, BB = _convert_to_blas_type(A, B)
    ldiv!(transpose(lu(AA)), BB) # call BlasFloat versions
end

function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::AbstractVecOrMat{S}) where {T<:Number, S<:Number}
    A = parent(Ac)
    checksquare(A)
    AA, BB = _convert_to_blas_type(A, B)
    ldiv!(adjoint(lu(AA)), BB) # call BlasFloat versions
end

function ldiv!(A::BandedLU{T}, B::AbstractVecOrMat{T}) where {T<:BlasFloat}
    checksquare(A)
    gbtrs!('N', A.m, A.l, A.u, A.data, A.ipiv, B)
end

function ldiv!(At::Transpose{T,BandedLU{T}}, B::AbstractVecOrMat{T}) where {T<:BlasFloat}
    A = parent(At)
    checksquare(A)
    gbtrs!('T', A.m, A.l, A.u, A.data, A.ipiv, B)
end
function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::AbstractVecOrMat{T}) where {T<:BlasReal}
    A = parent(Ac)
    checksquare(A)
    gbtrs!('T', A.m, A.l, A.u, A.data, A.ipiv, B)
end
function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::AbstractVecOrMat{T}) where {T<:BlasComplex}
    A = parent(Ac)
    checksquare(A)
    gbtrs!('C', A.m, A.l, A.u, A.data, A.ipiv, B)
end


factorize(A::BandedMatrix) = lu(A)
