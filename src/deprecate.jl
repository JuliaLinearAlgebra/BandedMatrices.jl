@deprecate gbmm!(α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T<:BlasFloat} gbmm!('N', 'N', α, A, B, β, C)
