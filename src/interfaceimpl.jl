

# Here we implement the banded matrix interface for some key examples
isbanded(::Zeros) = true
bandwidths(::Zeros) = (0,0)
inbands_getindex(::Zeros{T}, k::Integer, j::Integer) where T = zero(T)

isbanded(::Eye) = true
bandwidths(::Eye) = (0,0)
inbands_getindex(::Eye{T}, k::Integer, j::Integer) where T = one(T)

isbanded(::Diagonal) = true
bandwidths(::Diagonal) = (0,0)
inbands_getindex(D::Diagonal, k::Integer, j::Integer) = D.diag[k]
inbands_setindex!(D::Diagonal, v, k::Integer, j::Integer) = (D.diag[k] = v)

isbanded(::SymTridiagonal) = true
bandwidths(::SymTridiagonal) = (1,1)
inbands_getindex(J::SymTridiagonal, k::Integer, j::Integer) =
    k == j ? J.dv[k] : J.ev[k]
inbands_setindex!(J::SymTridiagonal, v, k::Integer, j::Integer) =
    k == j ? (J.dv[k] = v) : (J.ev[k] = v)

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.arrays)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.arrays
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end
kron(A::AbstractBandedMatrix, B::AbstractBandedMatrix) = BandedMatrix(Kron(A,B))
kron(A::AdjOrTrans{<:Any,<:AbstractBandedMatrix}, B::AbstractBandedMatrix) = BandedMatrix(Kron(A,B))
kron(A::AbstractBandedMatrix, B::AdjOrTrans{<:Any,<:AbstractBandedMatrix}) = BandedMatrix(Kron(A,B))
kron(A::AdjOrTrans{<:Any,<:AbstractBandedMatrix}, B::AdjOrTrans{<:Any,<:AbstractBandedMatrix}) = BandedMatrix(Kron(A,B))
