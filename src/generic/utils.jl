# BLAS/linear algebra overrides

@inline dot(x...) = LinearAlgebra.dot(x...)

dotu(f::StridedVector{T},g::StridedVector{T}) where {T<:Union{ComplexF32,ComplexF64}} =
    BLAS.dotu(f,g)
dotu(f::AbstractVector{Complex{Float64}},g::AbstractVector{N}) where {N<:Real} = dot(conj(f),g)
dotu(f::AbstractVector{N},g::AbstractVector{T}) where {N<:Real,T<:Number} = dot(f,g)


normalize!(w::AbstractVector) = rmul!(w,inv(norm(w)))
normalize!(w::Vector{T}) where {T<:BlasFloat} = normalize!(length(w),w)


# check dimensions of inputs
checkdimensions(sizedest::Tuple{Int, Vararg{Int}}, sizesrc::Tuple{Int, Vararg{Int}}) =
    (sizedest == sizesrc ||
        throw(DimensionMismatch("tried to assign $(sizesrc) sized " *
                                "array to $(sizedest) destination")) )

checkdimensions(dest::AbstractVector, src::AbstractVector) =
    checkdimensions(size(dest), size(src))

checkdimensions(ldest::Int, src::AbstractVector) =
    checkdimensions((ldest, ), size(src))

checkdimensions(kr::AbstractRange, jr::AbstractRange, src::AbstractMatrix) =
    checkdimensions((length(kr), length(jr)), size(src))


# return the bandwidths of A*B
prodbandwidths(A) = bandwidths(A)
prodbandwidths() = (0,0)
prodbandwidths(A...) = broadcast(+, bandwidths.(A)...)

# helper functions in matrix addition routines
function sumbandwidths(A::AbstractMatrix, B::AbstractMatrix)
    max(bandwidth(A, 1), bandwidth(B, 1)), max(bandwidth(A, 2), bandwidth(B, 2))
end
