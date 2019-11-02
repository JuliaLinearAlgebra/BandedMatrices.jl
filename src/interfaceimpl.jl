

##
# Sparse BroadcastStyle
##

for Typ in (:Diagonal, :SymTridiagonal, :Tridiagonal)
    @eval begin
        BroadcastStyle(::StructuredMatrixStyle{<:$Typ}, ::BandedStyle) =
            BandedStyle()
        BroadcastStyle(::BandedStyle, ::StructuredMatrixStyle{<:$Typ}) =
            BandedStyle()
    end
end

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
bandeddata(D::Diagonal) = reshape(D.diag, 1, length(D.diag))

# treat subinds as banded
sublayout(::DiagonalLayout{L}, inds::Type) where L =
    sublayout(bandedcolumns(L()), inds)

# bandeddata(V::SubArray{<:Any,2,<:Diagonal}) = view(bandeddata(parent(V)), :, parentindices(V)[2])

isbanded(::SymTridiagonal) = true
bandwidths(::SymTridiagonal) = (1,1)
inbands_getindex(J::SymTridiagonal, k::Integer, j::Integer) =
    k == j ? J.dv[k] : J.ev[min(k,j)]
inbands_setindex!(J::SymTridiagonal, v, k::Integer, j::Integer) =
    k == j ? (J.dv[k] = v) : (J.ev[min(k,j)] = v)

isbanded(::Tridiagonal) = true
bandwidths(::Tridiagonal) = (1,1)

