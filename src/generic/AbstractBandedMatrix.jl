# AbstractBandedMatrix must implement

abstract type AbstractBandedMatrix{T} <: AbstractSparseMatrix{T,Int} end

@doc """
    bandwidths(A)

Returns a tuple containing the upper and lower bandwidth of `A`.
"""
bandwidths(A::AbstractVecOrMat) = bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractVecOrMat) = -bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractVecOrMat, k::Integer) = k==1 ? -bandwidth(A,1) : bandwidth(A,2)


@doc """
    bandwidth(A,i)

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).
"""
bandwidth(A::AbstractVecOrMat, k::Integer) = k==1 ? size(A,1)-1 : size(A,2)-1

@doc """
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


@doc """
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


## Show

struct PrintShow
    str
end
Base.show(io::IO,N::PrintShow) = print(io,N.str)

if VERSION < v"0.7-"
    showarray(io,M;opts...) = Base.showarray(io,M,false;opts...)
    function Base.showarray(io::IO,B::AbstractBandedMatrix,repr::Bool = true; header = true)
        header && print(io,summary(B))

        if !isempty(B) && size(B,1) ≤ 1000 && size(B,2) ≤ 1000
            header && println(io,":")
            M=Array{Any}(size(B)...)
            fill!(M,PrintShow(""))
            for j = 1:size(B,2), k = colrange(B,j)
                M[k,j]=B[k,j]
            end

            showarray(io,M;header=false)
        end
    end
end
