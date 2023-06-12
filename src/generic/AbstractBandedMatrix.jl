# AbstractBandedMatrix must implement

abstract type AbstractBandedMatrix{T} <: LayoutMatrix{T} end


"""
    bandeddata(A)

returns a matrix containing the data of a banded matrix, in the
BLAS format.

This is required for gbmv! support
"""
bandeddata(A) = bandeddata(MemoryLayout(A), A)
bandeddata(_, A) = error("Override bandeddata(::$(typeof(A)))")

"""
    bandwidths(A)

Returns a tuple containing the lower and upper bandwidth of `A`, in order.
"""
bandwidths(A::AbstractVecOrMat) = bandwidths(MemoryLayout(A), A)
bandwidths(_, A) = (size(A,1)-1 , size(A,2)-1)

function BandedMatrices.bandwidths(A::SparseMatrixCSC)
    l,u = -size(A,1),-size(A,2)

    m,n = size(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    for j = 1:n
        for ind in nzrange(A, j)
            i = rows[ind]
            # We skip non-structural zeros when computing the
            # bandwidths.
            iszero(vals[ind]) && continue
            ij = abs(i-j)
            if i ≥ j
                l = max(l, ij)
                u = max(u, -ij)
            elseif i < j
                l = max(l, -ij)
                u = max(u, ij)
            end
        end
    end

    l,u
end

"""
    bandwidth(A,i)

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).
"""
bandwidth(A, k::Integer) = bandwidths(A)[k]

"""
    bandrange(A)

Returns the range `-bandwidth(A,1):bandwidth(A,2)`.
"""
bandrange(A) = -bandwidth(A,1):bandwidth(A,2)



"""
    colstart(A, i::Integer)

Return the starting row index of the filled bands in the i-th column,
bounded by the actual matrix size.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.colstart(A, 3)
2

julia> BandedMatrices.colstart(A, 4)
3
```
"""
@inline colstart(A, i::Integer) = max(i-bandwidth(A,2), 1)

"""
    colstop(A, i::Integer)

Return the stopping row index of the filled bands in the i-th column,
bounded by the actual matrix size.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.colstop(A, 3)
3

julia> BandedMatrices.colstop(A, 4)
4
```
"""
@inline  colstop(A, i::Integer) = max(min(i+bandwidth(A,1), size(A, 1)), 0)

"""
    rowstart(A, i::Integer)

Return the starting column index of the filled bands in the i-th row,
bounded by the actual matrix size.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.rowstart(A, 2)
2

julia> BandedMatrices.rowstart(A, 3)
3
```
"""
@inline rowstart(A, i::Integer) = max(i-bandwidth(A,1), 1)

"""
    rowstop(A, i::Integer)

Return the stopping column index of the filled bands in the i-th row,
bounded by the actual matrix size.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.rowstop(A, 2)
3

julia> BandedMatrices.rowstop(A, 4)
4
```
"""
@inline  rowstop(A, i::Integer) = max(min(i+bandwidth(A,2), size(A, 2)), 0)

"""
    colrange(A, i::Integer)

Return the range of rows in the `i`-th column that correspond to filled bands.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> colrange(A, 1)
1:1

julia> colrange(A, 3)
2:3
```
"""
@inline colrange(A, i::Integer) = colstart(A,i):colstop(A,i)

"""
    rowrange(A, i::Integer)

Return the range of columns in the `i`-th row that correspond to filled bands.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> rowrange(A, 1)
1:2

julia> rowrange(A, 4)
4:4
```
"""
@inline rowrange(A, i::Integer) = rowstart(A,i):rowstop(A,i)


"""
    collength(A, i::Integer)

Return the number of filled bands in the `i`-th column.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.collength(A, 1)
1

julia> BandedMatrices.collength(A, 2)
2
```
"""
@inline collength(A, i::Integer) = max(colstop(A, i) - colstart(A, i) + 1, 0)

"""
    rowlength(A, i::Integer)

Return the number of filled bands in the `i`-th row.

# Examples
```jldoctest
julia> A = BandedMatrix(0=>1:4, 1=>5:7)
4×4 BandedMatrix{Int64} with bandwidths (0, 1):
 1  5  ⋅  ⋅
 ⋅  2  6  ⋅
 ⋅  ⋅  3  7
 ⋅  ⋅  ⋅  4

julia> BandedMatrices.rowlength(A, 1)
2

julia> BandedMatrices.rowlength(A, 4)
1
```
"""
@inline rowlength(A, i::Integer) = max(rowstop(A, i) - rowstart(A, i) + 1, 0)

@inline banded_colsupport(A, j::Integer) = colrange(A, j)
@inline banded_rowsupport(A, j::Integer) = rowrange(A, j)

@inline banded_rowsupport(A, j) = isempty(j) ? (1:0) : rowstart(A,minimum(j)):rowstop(A,maximum(j))
@inline banded_colsupport(A, j) = isempty(j) ? (1:0) : colstart(A,minimum(j)):colstop(A,maximum(j))

@inline colsupport(::AbstractBandedLayout, A, j) = banded_colsupport(A, j)
@inline rowsupport(::AbstractBandedLayout, A, j) = banded_rowsupport(A, j)

"""
    isbanded(A)

returns true if a matrix implements the banded interface.
"""
isbanded(A) = isbanded(MemoryLayout(A), A)
isbanded(::AbstractBandedLayout, A) = true
isbanded(@nospecialize(::Any), @nospecialize(::Any)) = false

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

function tril!(A::AbstractBandedMatrix{T}, k::Integer) where T
    for b = max(k+1,-bandwidth(A,1)):bandwidth(A,2)
        A[band(b)] .= zero(T)
    end
    A
end

function triu!(A::AbstractBandedMatrix{T}, k::Integer) where T
    for b = -bandwidth(A,1):min(k-1,bandwidth(A,2))
        A[band(b)] .= zero(T)
    end
    A
end

function isdiag(B::AbstractBandedMatrix)
    l, u = bandwidths(B)
    l + u < 0 || (iszero(l) && iszero(u))
end
function istriu(B::AbstractBandedMatrix, k::Integer)
    l, u = bandwidths(B)
    l + u < 0 || min(l, size(B,1)-1) <= -k
end
function istril(B::AbstractBandedMatrix, k::Integer)
    l, u = bandwidths(B)
    l + u < 0 || min(u, size(B,2)-1) <= k
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


####
# AdjOrTrans
####

bandwidths(A::AdjOrTrans{T,S}) where {T,S} = reverse(bandwidths(parent(A)))
if VERSION >= v"1.9"
    copy(A::Adjoint{T,<:AbstractBandedMatrix}) where T = copy(parent(A))'
    copy(A::Transpose{T,<:AbstractBandedMatrix}) where T = transpose(copy(parent(A)))
end
