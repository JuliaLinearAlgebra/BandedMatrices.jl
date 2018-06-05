@_banded_linalg AbstractBandedMatrix


if VERSION < v"0.7-"
    ## Method definitions for generic eltypes - will make copies

    # Direct and transposed algorithms
    for typ in [BandedMatrix, BandedLU]
        for fun in [:A_ldiv_B!, :At_ldiv_B!]
            @eval function $fun(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
                checksquare(A)
                AA, BB = _convert_to_blas_type(A, B)
                $fun(lufact(AA), BB) # call BlasFloat versions
            end
        end
        # \ is different because it needs a copy, but we have to avoid ambiguity
        @eval function \(A::$typ{T}, B::VecOrMat{Complex{T}}) where {T<:BlasReal}
            checksquare(A)
            A_ldiv_B!(convert($typ{Complex{T}}, A), copy(B)) # goes to BlasFloat call
        end
        @eval function \(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
            checksquare(A)
            TS = _promote_to_blas_type(T, S)
            A_ldiv_B!(convert($typ{TS}, A), copy_oftype(B, TS)) # goes to BlasFloat call
        end
    end

    # Hermitian conjugate
    for typ in [BandedMatrix, BandedLU]
        @eval function Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Complex, S<:Number}
            checksquare(A)
            AA, BB = _convert_to_blas_type(A, B)
            Ac_ldiv_B!(lufact(AA), BB) # call BlasFloat versions
        end
        @eval Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Real, S<:Real} =
            At_ldiv_B!(A, B)
        @eval Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Real, S<:Complex} =
            At_ldiv_B!(A, B)
    end


    # Method definitions for BlasFloat types - no copies

    # Direct and transposed algorithms
    for (ch, fname) in zip(('N', 'T'), (:A_ldiv_B!, :At_ldiv_B!))
        # provide A*_ldiv_B!(::BandedLU, ::StridedVecOrMat) for performance
        @eval function $fname(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
            checksquare(A)
            gbtrs!($ch, A.l, A.u, A.m, A.data, A.ipiv, B)
        end
        # provide A*_ldiv_B!(::BandedMatrix, ::StridedVecOrMat) for generality
        @eval function $fname(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
            checksquare(A)
            $fname(lufact(A), B)
        end
    end

    # Hermitian conjugate algorithms - same two routines as above
    function Ac_ldiv_B!(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
        checksquare(A)
        gbtrs!('C', A.l, A.u, A.m, A.data, A.ipiv, B)
    end

    function Ac_ldiv_B!(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
        checksquare(A)
        Ac_ldiv_B!(lufact(A), B)
    end

    # fall back for real inputs
    Ac_ldiv_B!(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasReal} =
        At_ldiv_B!(A, B)
    Ac_ldiv_B!(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasReal} =
        At_ldiv_B!(A, B)
else
    ## Method definitions for generic eltypes - will make copies

    # Direct and transposed algorithms

    function ldiv!(A::BandedMatrix{T,Matrix{T}}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(lufact(AA), BB) # call BlasFloat versions
    end

    function ldiv!(At::Transpose{T,BandedMatrix{T,Matrix{T}}}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        A = parent(At)
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(transpose(lufact(AA)), BB) # call BlasFloat versions
    end

    function ldiv!(Ac::Adjoint{T,BandedMatrix{T,Matrix{T}}}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        A = parent(Ac)
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(adjoint(lufact(AA)), BB) # call BlasFloat versions
    end

    function ldiv!(A::BandedLU{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(lufact(AA), BB) # call BlasFloat versions
    end

    function ldiv!(At::Transpose{T,BandedLU{T}}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        A = parent(At)
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(transpose(lufact(AA)), BB) # call BlasFloat versions
    end

    function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        A = parent(Ac)
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        ldiv!(adjoint(lufact(AA)), BB) # call BlasFloat versions
    end



# \ is different because it needs a copy, but we have to avoid ambiguity
    function \(A::BandedLU{T}, B::VecOrMat{Complex{T}}) where {T<:BlasReal}
        checksquare(A)
        ldiv!(convert(BandedLU{Complex{T}}, A), copy(B)) # goes to BlasFloat call
    end
    function \(A::BandedLU{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        checksquare(A)
        TS = _promote_to_blas_type(T, S)
        ldiv!(convert(BandedLU{TS}, A), copy_oftype(B, TS)) # goes to BlasFloat call
    end

    # Method definitions for BlasFloat types - no copies
    @eval function ldiv!(A::BandedMatrix{T,Matrix{T}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        ldiv!(lufact(A), B)
    end
    @eval function ldiv!(A::Transpose{T,BandedMatrix{T,Matrix{T}}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        ldiv!(transpose(lufact(parent(A))), B)
    end
    @eval function ldiv!(A::Adjoint{T,BandedMatrix{T,Matrix{T}}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        ldiv!(adjoint(lufact(parent(A))), B)
    end



    # Direct and transposed algorithms
    function ldiv!(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        gbtrs!('N', A.l, A.u, A.m, A.data, A.ipiv, B)
    end

    function ldiv!(At::Transpose{T,BandedLU{T}}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        A = parent(At)
        checksquare(A)
        gbtrs!('T', A.l, A.u, A.m, A.data, A.ipiv, B)
    end
    function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::StridedVecOrMat{T}) where {T<:BlasReal}
        A = parent(Ac)
        checksquare(A)
        gbtrs!('T', A.l, A.u, A.m, A.data, A.ipiv, B)
    end
    function ldiv!(Ac::Adjoint{T,BandedLU{T}}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
        A = parent(Ac)
        checksquare(A)
        gbtrs!('C', A.l, A.u, A.m, A.data, A.ipiv, B)
    end
end
