

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

isbanded(::SymTridiagonal) = true
bandwidths(::SymTridiagonal) = (1,1)
inbands_getindex(J::SymTridiagonal, k::Integer, j::Integer) =
    k == j ? J.dv[k] : J.ev[min(k,j)]
inbands_setindex!(J::SymTridiagonal, v, k::Integer, j::Integer) =
    k == j ? (J.dv[k] = v) : (J.ev[min(k,j)] = v)

isbanded(::Tridiagonal) = true
bandwidths(::Tridiagonal) = (1,1)

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.arrays)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.arrays
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(AbstractTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}))

const OtherBandedMatrixTypes = (:Zeros, :Eye, :Diagonal, :SymTridiagonal)

for T1 in BandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in BandedMatrixTypes, T2 in OtherBandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in OtherBandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end



###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###

function _copyto!(::VcatLayout{<:Tuple{<:Any,ZerosLayout}}, y::AbstractVector,
                 M::MatMulVec{<:AbstractBandedLayout,<:VcatLayout{<:Tuple{<:Any,ZerosLayout}}})
    A,x = M.args
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    ỹ,_ = y.arrays
    x̃,_ = x.arrays

    length(ỹ) ≥ min(length(M),length(x̃)+bandwidth(A,1)) ||
        throw(InexactError("Cannot assign non-zero entries to Zero"))

    ỹ .= Mul(view(A, axes(ỹ,1), axes(x̃,1)) , x̃)
    y
end

function similar(M::MatMulVec{<:AbstractBandedLayout,<:VcatLayout{<:Tuple{<:Any,ZerosLayout}}}, ::Type{T}) where T
    A,x = M.args
    xf,_ = x.arrays
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end


###
# MulMatrix
###

bandwidths(M::MulMatrix) = bandwidths(M.applied)
isbanded(M::MulMatrix) = all(isbanded, M.applied.args)

const MulBandedMatrix{T} = MulMatrix{T, <:Mul{<:LayoutApplyStyle{<:Tuple{Vararg{<:AbstractBandedLayout}}}}}

BroadcastStyle(::Type{<:MulBandedMatrix}) = BandedStyle()

Base.replace_in_print_matrix(A::MulBandedMatrix, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)

function _banded_mul_getindex(::Type{T}, (A, B), k::Integer, j::Integer) where T
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    n = size(A,2)
    ret = zero(T)
    for ν = max(1,j-Bu,k-Al):min(n,j+Bl,k+Au)
        ret += A[k,ν] * B[ν,j]
    end
    ret
end


getindex(M::MatMulMat{<:AbstractBandedLayout,<:AbstractBandedLayout}, k::Integer, j::Integer) =
    _banded_mul_getindex(eltype(M), M.args, k, j)

getindex(M::Mul{<:LayoutApplyStyle{<:Tuple{Vararg{<:AbstractBandedLayout}}}}, k::Integer, j::Integer) =
    _banded_mul_getindex(eltype(M), (first(M.args), Mul(tail(M.args)...)), k, j)


@inline _sub_materialize(::MulLayout{<:Tuple{Vararg{<:AbstractBandedLayout}}}, V) = BandedMatrix(V)

MemoryLayout(V::SubArray{T,2,<:MulBandedMatrix,I}) where {T,I<:Tuple{Vararg{AbstractUnitRange}}} =
    MemoryLayout(parent(V))

@inline getindex(A::MulBandedMatrix, kr::Colon, jr::Colon) = _lazy_getindex(A, kr, jr)
@inline getindex(A::MulBandedMatrix, kr::Colon, jr::AbstractUnitRange) = _lazy_getindex(A, kr, jr)
@inline getindex(A::MulBandedMatrix, kr::AbstractUnitRange, jr::Colon) = _lazy_getindex(A, kr, jr)
@inline getindex(A::MulBandedMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = _lazy_getindex(A, kr, jr)
