

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

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.arrays)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.arrays
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(AbstractTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}),
                                    :Zeros, :Eye, :Diagonal, :SymTridiagonal)

for T1 in BandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###

function _copyto!(::VcatLayout{<:Tuple{<:Any,ZerosLayout}}, y::AbstractVector,
                 M::MatMulVec{<:BandedColumnMajor,<:VcatLayout{<:Tuple{<:Any,ZerosLayout}}})
    A,x = M.factors
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    ỹ,_ = y.arrays
    x̃,_ = x.arrays

    length(ỹ) ≥ length(x̃)+bandwidth(A,1) || throw(InexactError("Cannot assign non-zero entries to Zero"))

    @time ỹ .= Mul(view(A, axes(ỹ,1), axes(x̃,1)) , x̃)
    y
end

function similar(M::MatMulVec{<:BandedColumnMajor,<:VcatLayout{<:Tuple{<:Any,ZerosLayout}}}, ::Type{T}) where T
    A,x = M.factors
    xf,_ = x.arrays
    n = max(0,length(xf) + bandwidth(A,1))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end
