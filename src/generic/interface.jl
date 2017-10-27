# these are the routines of the banded interface of other AbstractMatrices

banded_axpy!(a::Number, X::AbstractMatrix{U}, Y::AbstractMatrix{V}) where {U, V} = banded_generic_axpy!(a, X, Y)

# matrix * vector

# for AbstractMatrix, uses the generic version of multiplication
banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V} = banded_generic_matvecmul!(c, tA, A, b)
banded_A_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'N', A, b)
banded_Ac_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'C', A, b)
banded_At_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'T', A, b)


# matrix * matrix

banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V} = banded_generic_matmatmul!(C, tA, tB, A, B)
banded_A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
banded_Ac_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'N', A, B)
banded_At_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'N', A, B)
banded_A_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'C', A, B)
banded_A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'T', A, B)
banded_Ac_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'C', A, B)
banded_At_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'T', A, B)
