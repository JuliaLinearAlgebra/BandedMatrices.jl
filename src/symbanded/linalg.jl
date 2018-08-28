symmetriclayout(layout::BandedColumnMajor, uplo) = SymmetricLayout(layout,uplo)
symmetriclayout(layout::BandedRowMajor, uplo) = SymmetricLayout(layout,uplo)


banded_sbmv!(α::T, A::AbstractMatrix{T}, x::AbstractVector{T}, β::T, y::AbstractVector{T}) where {T<:BlasFloat} =
  sbmv!('U', A.k, α, bandeddata(A), x, β, y)
