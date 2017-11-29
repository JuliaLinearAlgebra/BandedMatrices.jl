

# Here we implement the banded matrix interface for some key examples
isbanded(::Zeros) = true
bandwidth(::Zeros, k::Integer) = 0
inbands_getindex(::Zeros{T}, k::Integer, j::Integer) where T = zero(T)

isbanded(::Eye) = true
bandwidth(::Eye, k::Integer) = 0
inbands_getindex(::Eye{T}, k::Integer, j::Integer) where T = one(T)

isbanded(::Diagonal) = true
bandwidth(::Diagonal, k::Integer) = 0
inbands_getindex(D::Diagonal, k::Integer, j::Integer) = D.diag[k]
inbands_setindex!(D::Diagonal, v, k::Integer, j::Integer) = (D.diag[k] = v)

isbanded(::SymTridiagonal) = true
bandwidth(::SymTridiagonal, k::Integer) = 1
inbands_getindex(J::SymTridiagonal, k::Integer, j::Integer) =
    k == j ? J.dv[k] : J.ev[k]
inbands_setindex!(J::SymTridiagonal, v, k::Integer, j::Integer) =
    k == j ? (J.dv[k] = v) : (J.ev[k] = v)
