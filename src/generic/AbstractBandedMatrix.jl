# AbstractBandedMatrix must implement

abstract type AbstractBandedMatrix{T} <: AbstractSparseMatrix{T,Int} end


"""
    bandeddata(A)

returns a matrix containing the data of a banded matrix, in the
BLAS format.

This is required for gbmv! support
"""
bandeddata(A) = error("Override bandeddata(::$(typeof(A)))")

"""
    bandwidths(A)

Returns a tuple containing the upper and lower bandwidth of `A`.
"""
bandwidths(A::AbstractVecOrMat) = bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractVecOrMat) = -bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractVecOrMat, k::Integer) = k==1 ? -bandwidth(A,1) : bandwidth(A,2)

bandwidths(A::AdjOrTrans{T,S}) where {T,S} = reverse(bandwidths(parent(A)))
bandwidth(A::AdjOrTrans{T,S}, i::Int) where {T,S} = bandwidths(A)[i]


"""
    bandwidth(A,i)

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).
"""
bandwidth(A::AbstractVecOrMat, k::Integer) = k==1 ? size(A,1)-1 : size(A,2)-1

"""
    bandrange(A)

Returns the range `-bandwidth(A,1):bandwidth(A,2)`.
"""
bandrange(A::AbstractVecOrMat) = -bandwidth(A,1):bandwidth(A,2)



# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline colstart(A::AbstractVecOrMat, i::Integer) = max(i-bandwidth(A,2), 1)
@inline  colstop(A::AbstractVecOrMat, i::Integer) = max(min(i+bandwidth(A,1), size(A, 1)), 0)
@inline rowstart(A::AbstractVecOrMat, i::Integer) = max(i-bandwidth(A,1), 1)
@inline  rowstop(A::AbstractVecOrMat, i::Integer) = max(min(i+bandwidth(A,2), size(A, 2)), 0)


@inline colrange(A::AbstractVecOrMat, i::Integer) = colstart(A,i):colstop(A,i)
@inline rowrange(A::AbstractVecOrMat, i::Integer) = rowstart(A,i):rowstop(A,i)


# length of i-the column/row
@inline collength(A::AbstractVecOrMat, i::Integer) = max(colstop(A, i) - colstart(A, i) + 1, 0)
@inline rowlength(A::AbstractVecOrMat, i::Integer) = max(rowstop(A, i) - rowstart(A, i) + 1, 0)


"""
    isbanded(A)

returns true if a matrix implements the banded interface.
"""
isbanded(::AbstractBandedMatrix) = true
isbanded(_) = false

# override bandwidth(A,k) for each AbstractBandedMatrix
# override inbands_getindex(A,k,j)


# return id of first empty diagonal intersected along row k
function _firstdiagrow(A::AbstractMatrix, k::Int)
    a, b = rowstart(A, k), rowstop(A, k)
    c = a == 1 ? b+1 : a-1
    c-k
end

# return id of first empty diagonal intersected along column j
function _firstdiagcol(A::AbstractMatrix, j::Int)
    a, b = colstart(A, j), colstop(A, j)
    r = a == 1 ? b+1 : a-1
    j-r
end

function Base.maximum(B::AbstractBandedMatrix)
    m=zero(eltype(B))
    for j = 1:size(B,2), k = colrange(B,j)
        m=max(B[k,j],m)
    end
    m
end

# fallbacks for inbands_getindex and inbands_setindex!
@inline function inbands_getindex(x::AbstractMatrix, i::Integer, j::Integer)
    @inbounds r = getindex(x, i, j)
    r
end
@inline function inbands_setindex!(x::AbstractMatrix, v, i::Integer, j::Integer)
    @inbounds r = setindex!(x, v, i, j)
    r
end

inbands_getindex(x::Adjoint, i::Integer, j::Integer) =
    inbands_getindex(parent(x), j, i)'
inbands_getindex(x::Transpose, i::Integer, j::Integer) =
    transpose(inbands_getindex(parent(x), j, i))
inbands_setindex!(x::Adjoint, v, i::Integer, j::Integer) =
    inbands_setindex!(parent(x), v', j, i)
inbands_setindex!(x::Transpose, v, i::Integer, j::Integer) =
    inbands_setindex!(parent(x), transpose(v), j, i)
## Show


## structured matrix methods ##
function Base.replace_in_print_matrix(A::AbstractBandedMatrix, i::Integer, j::Integer, s::AbstractString)
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)
end


## @inbands

function inbands_process_args!(e)
    if e.head == :ref
        e.head = :call
        pushfirst!(e.args, :(BandedMatrices.inbands_getindex))
    elseif e.head == :call && e.args[1] == :getindex
        e.args[1] = :(BandedMatrices.inbands_getindex)
    end
    e
end


function inbands_process_args_recursive!(expr)
    for (i,e) in enumerate(expr.args)
        if e isa Expr
            inbands_process_args_recursive!(e)
        end
    end
    inbands_process_args!(expr)
    expr
end

macro inbands(expr)
    esc(inbands_process_args!(expr))
end
