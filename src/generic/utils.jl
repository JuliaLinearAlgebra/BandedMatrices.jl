
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


_fill_lmul!(β, A::AbstractArray, Azero=false) = iszero(β) ? (Azero ? A : zero!(A)) : lmul!(β, A)
_fill_rmul!(A::AbstractArray, β, Azero=false) = iszero(β) ? (Azero ? A : zero!(A)) : rmul!(A, β)
_fill_rmul!(M::MulAdd, β) = _fill_rmul!(M.C, β, M.Czero)
