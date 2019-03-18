
function banded_chol!(::BandedColumns{DenseColumnMajor}, 
                       A::AbstractMatrix{T}, ::Type{UpperTriangular}) where T<:BlasFloat
    C, info = pbtrf!('U', size(A, 1), bandwidth(A,2), bandeddata(A))
    UpperTriangular(C), info
end

## Non BLAS/LAPACK element types (generic)
function banded_chol!(_, A::AbstractMatrix, ::Type{UpperTriangular})
    @assert !has_offset_axes(A)
    n = checksquare(A)
    u = bandwidth(A,2)
    @inbounds begin
        for k = 1:n
            for i = max(1,k-u):k - 1
                A[k,k] -= A[i,k]'A[i,k]
            end
            Akk, info = _chol!(A[k,k], UpperTriangular)
            if info != 0
                return UpperTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(copy(Akk'))
            for j = k + 1:min(k+u,n)
                for i = max(1,j-u):k - 1
                    A[k,j] -= A[i,k]'A[i,j]
                end
                A[k,j] = AkkInv*A[k,j]
            end
        end
    end
    return UpperTriangular(A), convert(BlasInt, 0)
end
function banded_chol!(_, A::AbstractMatrix, ::Type{LowerTriangular})
    @assert !has_offset_axes(A)
    n = checksquare(A)
    l = bandwidth(A,1)
    @inbounds begin
        for k = 1:n
            for i = max(1,k-l):k - 1
                A[k,k] -= A[k,i]*A[k,i]'
            end
            Akk, info = _chol!(A[k,k], LowerTriangular)
            if info != 0
                return LowerTriangular(A), info
            end
            A[k,k] = Akk
            AkkInv = inv(Akk)
            for j = max(1,k-l):k - 1
                @simd for i = k + 1:min(n,j+l)
                    A[i,k] -= A[i,j]*A[k,j]'
                end
            end
            for i = k + 1:min(n,k+l)
                A[i,k] *= AkkInv'
            end
        end
     end
    return LowerTriangular(A), convert(BlasInt, 0)
end

banded_chol!(A, ::Type{T}) where T = banded_chol!(MemoryLayout(A), A, T)
chol!(A::AbstractBandedMatrix, ::Type{UpperTriangular}) = banded_chol!(A, UpperTriangular)
chol!(A::AbstractBandedMatrix, ::Type{LowerTriangular}) = banded_chol!(A, LowerTriangular)

_ldiv!(::BandedColumns{DenseColumnMajor}, A::Cholesky{T}, B::AbstractVecOrMat{T}) where T<:BlasFloat =
    pbtrs!('U', size(A, 1), bandwidth(A.factors,2), A.factors.data, B)

function _ldiv!(::AbstractBandedLayout, C::Cholesky, B::AbstractMatrix)
    if C.uplo == 'L'
        return ldiv!(adjoint(LowerTriangular(C.factors)), ldiv!(LowerTriangular(C.factors), B))
    else
        return ldiv!(UpperTriangular(C.factors), ldiv!(adjoint(UpperTriangular(C.factors)), B))
    end
end    

ldiv!(A::Cholesky{T,<:AbstractBandedMatrix}, B::StridedVecOrMat{T}) where T<:BlasFloat = 
    _ldiv!(MemoryLayout(A.factors), A, B)

# @lazyldiv Cholesky{<:Any,<:AbstractBandedMatrix}

# For some bizarre reason this isnt in LinearAlgebra
cholesky(A::Symmetric{T,<:BandedMatrix{T}},
    ::Val{false}=Val(false); check::Bool = true) where T<:Real = cholesky!(cholcopy(A); check = check)