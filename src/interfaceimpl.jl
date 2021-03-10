

##
# Sparse BroadcastStyle
##

for Typ in (:Diagonal, :SymTridiagonal, :Tridiagonal, :Bidiagonal)
    @eval begin
        BroadcastStyle(::StructuredMatrixStyle{<:$Typ}, ::BandedStyle) =
            BandedStyle()
        BroadcastStyle(::BandedStyle, ::StructuredMatrixStyle{<:$Typ}) =
            BandedStyle()
    end
end

# Here we implement the banded matrix interface for some key examples
isbanded(::Zeros) = true
bandwidths(::Zeros) = (-40320,-40320) # 40320 == prod(1:8), used for special cases involving gcd
inbands_getindex(::Zeros{T}, k::Integer, j::Integer) where T = zero(T)

isbanded(::Eye) = true
bandwidths(::Eye) = (0,0)
inbands_getindex(::Eye{T}, k::Integer, j::Integer) where T = one(T)

isbanded(::Diagonal) = true
bandwidths(::Diagonal) = (0,0)
inbands_getindex(D::Diagonal, k::Integer, j::Integer) = D.diag[k]
inbands_setindex!(D::Diagonal, v, k::Integer, j::Integer) = (D.diag[k] = v)
bandeddata(D::Diagonal) = permutedims(D.diag)

# treat subinds as banded
sublayout(::DiagonalLayout{L}, inds::Type) where L = sublayout(bandedcolumns(L()), inds)
sublayout(::DiagonalLayout{L}, inds::Type{<:NTuple{2,AbstractUnitRange{Int}}}) where L = sublayout(bandedcolumns(L()), inds)

# bandeddata(V::SubArray{<:Any,2,<:Diagonal}) = view(bandeddata(parent(V)), :, parentindices(V)[2])

isbanded(::Bidiagonal) = true
bandwidths(A::Bidiagonal) = A.uplo == 'U' ? (0,1) : (1,0)

sublayout(::BidiagonalLayout, ::Type{<:Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}}}) =
    BandedLayout()


isbanded(::SymTridiagonal) = true
bandwidths(::SymTridiagonal) = (1,1)
inbands_getindex(J::SymTridiagonal, k::Integer, j::Integer) =
    k == j ? J.dv[k] : J.ev[min(k,j)]
inbands_setindex!(J::SymTridiagonal, v, k::Integer, j::Integer) =
    k == j ? (J.dv[k] = v) : (J.ev[min(k,j)] = v)

isbanded(::Tridiagonal) = true
bandwidths(::Tridiagonal) = (1,1)

sublayout(::AbstractTridiagonalLayout, ::Type{<:Tuple{AbstractUnitRange{Int},AbstractUnitRange{Int}}}) =
    BandedLayout()

###
# rot180
###

function rot180(A::AbstractBandedMatrix)
    m,n = size(A)
    sh = m-n
    l,u = bandwidths(A)
    _BandedMatrix(bandeddata(A)[end:-1:1,end:-1:1], m, u+sh,l-sh)
end

function getindex(D::Diagonal{T,V}, b::Band) where {T,V}
    iszero(b.i) && return copy(D.diag)
    convert(V, Zeros{T}(size(D,1)-abs(b.i)))
end

function getindex(D::Tridiagonal{T,V}, b::Band) where {T,V}
    b.i == -1 && return copy(D.dl)
    iszero(b.i) && return copy(D.d)
    b.i == 1 && return copy(D.du)
    convert(V, Zeros{T}(size(D,1)-abs(b.i)))
end

function getindex(D::SymTridiagonal{T,V}, b::Band) where {T,V}
    iszero(b.i) && return copy(D.dv)
    abs(b.i) == 1 && return copy(D.ev)
    convert(V, Zeros{T}(size(D,1)-abs(b.i)))
end

function getindex(D::Bidiagonal{T,V}, b::Band) where {T,V}
    iszero(b.i) && return copy(D.dv)
    D.uplo == 'L' && b.i == -1 && return copy(D.ev)
    D.uplo == 'U' && b.i == 1 && return copy(D.ev)
    convert(V, Zeros{T}(size(D,1)-abs(b.i)))
end