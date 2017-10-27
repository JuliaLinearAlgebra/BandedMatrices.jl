
macro banded_banded_banded_linalg(Typ1, Typ2, Typ3)
    ret = quote
        BandedMatrices.positively_banded_matmatmul!(C::$Typ1{T}, tA::Char, tB::Char, A::$Typ2{T}, B::$Typ3{T}) where {T <: BlasFloat} =
            BandedMatrices._positively_banded_matmatmul!(C, tA, tB, A, B)
    end
    esc(ret)
end


# add banded linear algebra routines between Typ1 and Typ2 both implementing
# the BandedMatrix interface
macro banded_banded_linalg(Typ1, Typ2)
    ret = quote
        BandedMatrices.@banded_banded_banded_linalg($Typ2, $Typ1, $Typ2)  # the first argument is the destination

        BandedMatrices.banded_axpy!(a::Number, X::$Typ1, Y::$Typ2) = BandedMatrices.banded_generic_axpy!(a, X, Y)
        Base.BLAS.axpy!(a::Number, X::$Typ1, Y::$Typ2) = BandedMatrices.banded_axpy!(a, X, Y)

        function Base.:+(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
            n, m = size(A)
            ret = BandedMatrices.bzeros(promote_type(T,V), n, m, BandedMatrices.sumbandwidths(A, B)...)
            axpy!(one(T), A, ret)
            axpy!(one(V), B, ret)
            ret
        end

        function Base.:-(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
            n, m=size(A)
            ret = BandedMatrices.bzeros(promote_type(T,V), n, m, BandedMatrices.sumbandwidths(A, B)...)
            axpy!(one(T),  A, ret)
            axpy!(-one(V), B, ret)
            ret
        end

        BandedMatrices.banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::$Typ1{T}, B::$Typ2{T}) where {T <: BlasFloat} =
            BandedMatrices.generally_banded_matmatmul!(C, tA, tB, A, B)

        Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = banded_matmatmul!(C, 'N', 'N', A, B)

        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

        Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

        # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

        function Base.:*(A::$Typ1{T}, B::$Typ2{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            Y = BandedMatrix(promote_type(T,V), n, m, prodbandwidths(A, B)...)
            A_mul_B!(Y, A, B)
        end

        Base.LinAlg.Ac_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_B!(BandedMatrices.banded_similar('C', 'N', A, B, promote_type(T, V)), A, B)
        Base.LinAlg.A_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bc!(BandedMatrices.banded_similar('N', 'C', A, B, promote_type(T, V)), A, B)
        Base.LinAlg.Ac_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_Bc!(BandedMatrices.banded_similar('C', 'C', A, B, promote_type(T, V)), A, B)

        Base.LinAlg.At_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'N', A, B, promote_type(T, V)), A, B)
        Base.LinAlg.A_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bt!(BandedMatrices.banded_similar('N', 'T', A, B, promote_type(T, V)), A, B)
        Base.LinAlg.At_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'T', A, B, promote_type(T, V)), A, B)
    end
    esc(ret)
end

# add banded linear algebra routines for Typ implementing the BandedMatrix interface
macro banded_linalg(Typ)
    ret = quote
        BandedMatrices.@banded_banded_linalg($Typ, $Typ)
        # default is to use dense axpy!
        BandedMatrices.banded_axpy!(a::Number, X::$Typ, Y::AbstractMatrix) =
                BandedMatrices.banded_dense_axpy!(a, X, Y)
        Base.BLAS.axpy!(a::Number, X::$Typ, Y::AbstractMatrix) =
                BandedMatrices.banded_axpy!(a, X, Y)
        function Base.:+(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
            ret = deepcopy(B)
            axpy!(one(T), A, ret)
            ret
        end
        function Base.:+(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
            n, m=size(A)
            ret = zeros(promote_type(T,V),n,m)
            axpy!(one(T), A,ret)
            axpy!(one(V), B,ret)
            ret
        end
        Base.:+(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = B + A

        function Base.:-(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
            ret = deepcopy(B)
            Base.scale!(ret,-1)
            Base.BLAS.axpy!(one(T), A, ret)
            ret
        end
        function Base.:-(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
            n, m= size(A)
            ret = zeros(promote_type(T,V),n,m)
            Base.BLAS.axpy!(one(T),  A, ret)
            Base.BLAS.axpy!(-one(V), B, ret)
            ret
        end
        Base.:-(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = Base.scale!(B - A, -1)



        ## UniformScaling

        function Base.BLAS.axpy!(a::Number, X::UniformScaling, Y::$Typ{T}) where {T}
            BandedMatrices.checksquare(Y)
            α = a * X.λ
            @inbounds for k = 1:size(Y,1)
                BandedMatrices.inbands_setindex!(Y, BandedMatrices.inbands_getindex(Y, k, k) + α, k, k)
            end
            Y
        end

        function Base.:+(A::$Typ, B::UniformScaling)
            ret = deepcopy(A)
            Base.BLAS.axpy!(1, B,ret)
        end

        Base.:+(A::UniformScaling, B::$Typ) = B + A

        function Base.:-(A::$Typ, B::UniformScaling)
            ret = deepcopy(A)
            axpy!(-1, B, ret)
        end

        function Base.:-(A::UniformScaling, B::$Typ)
            ret = deepcopy(B)
            Base.scale!(ret, -1)
            axpy!(1, A, ret)
        end

        # use BLAS routine for positively banded BLASBandedMatrix
        BandedMatrices.positively_banded_matvecmul!(c::StridedVector{T}, tA::Char, A::$Typ{T}, b::StridedVector{T}) where {T <: BandedMatrices.BlasFloat} =
                BandedMatrices.gbmv!(tA, one(T), A, b, zero(T), c)


        BandedMatrices.banded_matvecmul!(c::StridedVector{T}, tA::Char, A::$Typ{T}, b::AbstractVector{T}) where {T <: BlasFloat} =
                BandedMatrices.generally_banded_matvecmul!(c, tA, A, b)

        Base.LinAlg.A_mul_B!(c::AbstractVector{T}, A::$Typ{U}, b::AbstractVector{V}) where {T, U, V} = BandedMatrices.banded_matvecmul!(c, 'N', A, b)
        Base.LinAlg.Ac_mul_B!(c::AbstractVector{T}, A::$Typ{U}, b::AbstractVector{V}) where {T, U, V} = BandedMatrices.banded_matvecmul!(c, 'C', A, b)
        Base.LinAlg.Ac_mul_B!(c::AbstractVector{T}, A::$Typ{U}, b::AbstractVector{V}) where {T, U<:Real, V} = BandedMatrices.banded_matvecmul!(c, 'T', A, b)
        Base.LinAlg.At_mul_B!(c::AbstractVector{T}, A::$Typ{U}, b::AbstractVector{V}) where {T, U, V} = BandedMatrices.banded_matvecmul!(c, 'T', A, b)

        Base.:*(A::$Typ{U}, b::StridedVector{V}) where {U, V} =
            Base.LinAlg.A_mul_B!(Vector{promote_type(U, V)}(size(A, 1)), A, b)

        BandedMatrices.positively_banded_matmatmul!(C::StridedMatrix{T}, tA::Char, tB::Char, A::$Typ{T}, B::StridedMatrix{T}) where {T <: BlasFloat} =
            BandedMatrices._positively_banded_matmatmul!(C, tA, tB, A, B)
        BandedMatrices.positively_banded_matmatmul!(C::StridedMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::$Typ{T}) where {T <: BlasFloat} =
            BandedMatrices._positively_banded_matmatmul!(C, tA, tB, A, B)

        BandedMatrices.banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::$Typ{T}, B::StridedMatrix{T}) where {T <: BlasFloat} =
            BandedMatrices.generally_banded_matmatmul!(C, tA, tB, A, B)
        BandedMatrices.banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::$Typ{T}) where {T <: BlasFloat} =
            BandedMatrices.generally_banded_matmatmul!(C, tA, tB, A, B)

        Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)
        Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)



        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

        Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
        Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

        # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
        Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)


        function Base.:*(A::$Typ{T}, B::StridedMatrix{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
        end

        function Base.:*(A::StridedMatrix{T}, B::$Typ{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
        end

        Base.LinAlg.Ac_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
        Base.LinAlg.Ac_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
        Base.LinAlg.A_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
        Base.LinAlg.A_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
        Base.LinAlg.Ac_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
        Base.LinAlg.Ac_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)

        Base.LinAlg.At_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
        Base.LinAlg.At_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
        Base.LinAlg.A_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
        Base.LinAlg.A_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
        Base.LinAlg.At_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
        Base.LinAlg.At_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
    end
    esc(ret)
end

@banded_linalg AbstractBandedMatrix
@banded_linalg BLASBandedMatrix # this supports views of banded matrices




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
    @eval function (\)(A::$typ{T}, B::VecOrMat{Complex{T}}) where {T<:BlasReal}
        checksquare(A)
        A_ldiv_B!(convert($typ{Complex{T}}, A), copy(B)) # goes to BlasFloat call
    end
    @eval function (\)(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
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
