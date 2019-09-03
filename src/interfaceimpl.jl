

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

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)
function bandwidths(K::Kron{<:Any,2})
    A,B = K.args
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

function similar(M::MulAdd{<:AbstractBandedLayout,<:ApplyLayout{typeof(vcat),<:Tuple{<:Any,ZerosLayout}}}, ::Type{T}, axes) where T
    A,x = M.A,M.B
    xf,_ = x.args
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end

function materialize!(M::MatMulVecAdd{<:AbstractBandedLayout,
                                    <:ApplyLayout{typeof(vcat),<:Tuple{<:Any,ZerosLayout}},
                                     <:ApplyLayout{typeof(vcat),<:Tuple{<:Any,ZerosLayout}}})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    ỹ,_ = y.args
    x̃,_ = x.args

    length(ỹ) ≥ min(length(M),length(x̃)+bandwidth(A,1)) ||
        throw(InexactError("Cannot assign non-zero entries to Zero"))

    materialize!(MulAdd(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, ỹ))
    y
end




###
# MulMatrix
###

bandwidths(M::MulMatrix) = bandwidths(Applied(M))
isbanded(M::Mul) = all(isbanded, M.args)
isbanded(M::MulMatrix) = isbanded(Applied(M))

const MulBandedLayout = MulLayout{<:Tuple{Vararg{<:AbstractBandedLayout}}}

applybroadcaststyle(::Type{<:AbstractMatrix}, ::MulBandedLayout) = BandedStyle()

@inline colsupport(::MulBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::MulBandedLayout, A, j) = banded_rowsupport(A, j)
@inline colsupport(::MulLayout{<:Tuple{<:AbstractBandedLayout,<:AbstractStridedLayout}}, A, j) = banded_colsupport(A, j)



@inline sub_materialize(::MulBandedLayout, V) = BandedMatrix(V)

subarraylayout(M::MulBandedLayout, ::Type{<:Tuple{Vararg{AbstractUnitRange}}}) = M



######
# Concat banded matrix
######

bandwidths(M::Hcat) = (bandwidth(M.args[1],1),sum(size.(M.args[1:end-1],2)) + bandwidth(M.args[end],2))
isbanded(M::Hcat) = all(isbanded, M.args)

bandwidths(M::Vcat) = (sum(size.(M.args[1:end-1],1)) + bandwidth(M.args[end],1), bandwidth(M.args[1],2))
isbanded(M::Vcat) = all(isbanded, M.args)


const HcatBandedMatrix{T,N} = Hcat{T,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}
const VcatBandedMatrix{T,N} = Vcat{T,2,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}

BroadcastStyle(::Type{HcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()
BroadcastStyle(::Type{VcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()

Base.replace_in_print_matrix(A::HcatBandedMatrix, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)
Base.replace_in_print_matrix(A::VcatBandedMatrix, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)    

hcat(A::BandedMatrix...) = BandedMatrix(Hcat(A...))    
hcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Hcat(A, B...))    

vcat(A::BandedMatrix...) = BandedMatrix(Vcat(A...))    
vcat(A::BandedMatrix, B::AbstractMatrix...) = Matrix(Vcat(A, B...))    



#######
# CachedArray
#######

bandwidths(B::CachedMatrix) = bandwidths(B.data)
isbanded(B::CachedMatrix) = isbanded(B.data)

function resizedata!(B::CachedMatrix{T,BandedMatrix{T,Matrix{T},OneTo{Int}}}, n::Integer, m::Integer) where T<:Number
    @boundscheck checkbounds(Bool, B, n, m) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = B.data
    ν,μ = size(olddata)
    n,m = max(ν,n), max(μ,m)
    if (ν,μ) ≠ (n,m)
        B.data = BandedMatrix{T}(undef, (n,m), bandwidths(olddata))
        B.data.data[:,1:μ] .= olddata.data
        bd = bandeddata(B.array)
        B.data.data[:,μ+1:end] .= view(bd,:,μ+1:m)
    end

    B.datasize = (n,m)

    B
end
