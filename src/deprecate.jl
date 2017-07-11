@deprecate gbmm!{T<:BlasFloat}(α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) gbmm!('N', 'N', α, A, B, β, C)
