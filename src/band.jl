# ~~ Type to set\get data along a band
immutable Band
    i::Int
end

doc"""
    band(i)

Represents the `i`-th band of a banded matrix.

```jldoctest
julia> using BandedMatrices

julia> A = bones(5,5,1,1)
5×5 BandedMatrices.BandedMatrix{Float64}:
 1.0  1.0
 1.0  1.0  1.0
      1.0  1.0  1.0
           1.0  1.0  1.0
                1.0  1.0

julia> A[band(1)]
4-element Array{Float64,1}:
 1.0
 1.0
 1.0
 1.0
```

"""
band(i::Int) = Band(i)


doc"""
    BandRange

Represents the entries in a row/column inside the bands.

```jldoctest
julia> using BandedMatrices

julia> A = bones(5,5,1,1)
5×5 BandedMatrices.BandedMatrix{Float64}:
 1.0  1.0
 1.0  1.0  1.0
      1.0  1.0  1.0
           1.0  1.0  1.0
                1.0  1.0

julia> A[2,BandRange]
3-element Array{Float64,1}:
 1.0
 1.0
 1.0
```
"""

# ~~ Indexing on the i-th row/column within band range
immutable BandRange end

# ~~ Out of band error
immutable BandError <: Exception
    A::AbstractBandedMatrix
    i::Int
end

function showerror(io::IO, e::BandError)
    A, i = e.A, e.i
    print(io, "attempt to access $(typeof(A)) with bandwidths " *
              "($(bandwidth(A, 1)), $(bandwidth(A, 2))) at band $i")
end

# start/stop indices of the i-th column/row, bounded by actual matrix size
@inline colstart(A, i::Integer) = max(i-bandwidth(A,2), 1)
@inline  colstop(A, i::Integer) = max(min(i+bandwidth(A,1), size(A, 1)), 0)
@inline rowstart(A, i::Integer) = max(i-bandwidth(A,1), 1)
@inline  rowstop(A, i::Integer) = max(min(i+bandwidth(A,2), size(A, 2)), 0)


@inline colrange(A, i::Integer) = colstart(A,i):colstop(A,i)
@inline rowrange(A, i::Integer) = rowstart(A,i):rowstop(A,i)

# length of i-the column/row
@inline collength(A, i::Integer) = max(colstop(A, i) - colstart(A, i) + 1, 0)
@inline rowlength(A, i::Integer) = max(rowstop(A, i) - rowstart(A, i) + 1, 0)

# length of diagonal
@inline diaglength(A::AbstractBandedMatrix, b::Band) = diaglength(A, b.i)
@inline function diaglength(A::AbstractBandedMatrix, i::Integer)
    max(min(size(A, 2), size(A, 1)+i) - max(0, i), 0)
end

# return id of first empty diagonal intersected along row k
function _firstdiagrow(A, k)
    a, b = rowstart(A, k), rowstop(A, k)
    c = a == 1 ? b+1 : a-1
    c-k
end

# return id of first empty diagonal intersected along column j
function _firstdiagcol(A, j)
    a, b = colstart(A, j), colstop(A, j)
    r = a == 1 ? b+1 : a-1
    j-r
end


# ~ bound checking functions ~

checkbounds(A::AbstractBandedMatrix, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractBandedMatrix, kr::Range, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractBandedMatrix, k::Integer, jr::Range) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractBandedMatrix, kr::Range, jr::Range) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractBandedMatrix, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractBandedMatrix, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))

# check indices fall in the band
checkband(A::AbstractBandedMatrix, i::Integer) =
    (bandinds(A, 1) ≤ i ≤ bandinds(A, 2) || throw(BandError(A, i)))

checkband(A::AbstractBandedMatrix, b::Band) = checkband(A, b.i)

checkband(A::AbstractBandedMatrix, k::Integer, j::Integer) = checkband(A, j-k)

checkband(A::AbstractBandedMatrix, kr::Range, j::Integer) =
    (checkband(A, first(kr), j); checkband(A,  last(kr), j))

checkband(A::AbstractBandedMatrix, k::Integer, jr::Range) =
    (checkband(A, k, first(jr)); checkband(A, k,  last(jr)))

checkband(A::AbstractBandedMatrix, kr::Range, jr::Range) =
    (checkband(A, kr, first(jr)); checkband(A, kr,  last(jr)))
