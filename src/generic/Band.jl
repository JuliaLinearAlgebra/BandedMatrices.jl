# ~~ Type to set\get data along a band
struct Band
    i::Int
end

show(io::IO, r::Band) = print(io, "Band(", r.i, ")")

"""
    band(i)

Represents the `i`-th band of a banded matrix.

```jldoctest
julia> using BandedMatrices

julia> A = BandedMatrix(0=>1:4, 1=>5:7, -1=>8:10)
4×4 BandedMatrix{Int64} with bandwidths (1, 1):
 1  5   ⋅  ⋅
 8  2   6  ⋅
 ⋅  9   3  7
 ⋅  ⋅  10  4

julia> A[band(1)]
3-element Vector{Int64}:
 5
 6
 7

julia> A[band(0)]
4-element Vector{Int64}:
 1
 2
 3
 4

julia> A[band(-1)]
3-element Vector{Int64}:
  8
  9
 10
```
"""
band(i::Int) = Band(i)


struct BandRangeType end
"""
    BandRange

Represents the entries in a row/column inside the bands.

```jldoctest
julia> using BandedMatrices

julia> A = BandedMatrix(0=>1:4, 1=>5:7, -1=>8:10)
4×4 BandedMatrix{Int64} with bandwidths (1, 1):
 1  5   ⋅  ⋅
 8  2   6  ⋅
 ⋅  9   3  7
 ⋅  ⋅  10  4

julia> A[2, BandRange]
3-element Vector{Int64}:
 8
 2
 6
```
"""
const BandRange = BandRangeType()

to_indices(A::AbstractArray, (_, j)::Tuple{BandRangeType,Integer}) = (colrange(A, j), j)
to_indices(A::AbstractArray, (k, _)::Tuple{Integer,BandRangeType}) = (k, rowrange(A, k))

# ~~ Out of band error
struct BandError <: Exception
    A::AbstractMatrix
    i::Int
end

# shorthand to specify k and j without calculating band
BandError(A::AbstractMatrix, (k, j)::Tuple{Int,Int}) = BandError(A, j-k)
BandError(A::AbstractMatrix) = BandError(A, max(size(A)...)-1)

function showerror(io::IO, e::BandError)
    A, i = e.A, e.i
    print(io, "BandError: attempt to access $(typeof(A)) with bandwidths " *
              "($(bandwidth(A, 1)), $(bandwidth(A, 2))) at band $i")
end


# length of diagonal
@inline diaglength(A::AbstractMatrix, b::Band) = diaglength(A, b.i)
@inline function diaglength(A::AbstractMatrix, i::Integer)
    max(min(size(A, 2), size(A, 1)+i) - max(0, i), 0)
end


# check indices fall in the band
checkband(A::AbstractMatrix, i::Integer) =
    (-bandwidth(A, 1) ≤ i ≤ bandwidth(A, 2) || throw(BandError(A, i)))

checkband(A::AbstractMatrix, b::Band) = checkband(A, b.i)

checkband(A::AbstractMatrix, k::Integer, j::Integer) = checkband(A, j-k)

checkband(A::AbstractMatrix, kr::AbstractRange, j::Integer) =
    (checkband(A, first(kr), j); checkband(A,  last(kr), j))

checkband(A::AbstractMatrix, k::Integer, jr::AbstractRange) =
    (checkband(A, k, first(jr)); checkband(A, k,  last(jr)))

checkband(A::AbstractMatrix, kr::AbstractRange, jr::AbstractRange) =
    (checkband(A, kr, first(jr)); checkband(A, kr,  last(jr)))


# checks if the bands match A
function checkbandmatch(A::AbstractMatrix{T}, V::AbstractVector, ::Colon, j::Integer) where {T}
    for k = 1:colstart(A,j)-1
        if V[k] ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
    for k = colstop(A,j)+1:size(A,1)
        if V[k] ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
end

function checkbandmatch(A::AbstractMatrix{T}, V::AbstractVector, kr::AbstractRange, j::Integer) where {T}
    a = colstart(A, j)
    b = colstop(A, j)
    i = 0
    for v in V
        k = kr[i+=1]
        if (k < a || k > b) && v ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
end

function checkbandmatch(A::AbstractMatrix{T}, V::AbstractVector, k::Integer, ::Colon) where {T}
    for j = 1:rowstart(A,k)-1
        if V[j] ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
    for j = rowstop(A,k)+1:size(A,2)
        if V[j] ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
end

function checkbandmatch(A::AbstractMatrix{T}, V::AbstractVector, k::Integer, jr::AbstractRange) where {T}
    a = rowstart(A, k)
    b = rowstop(A, k)
    i = 0
    for v in V
        j = jr[i+=1]
        if (j < a || j > b) && v ≠ zero(T)
            throw(BandError(A, (k,j)))
        end
    end
end

function checkbandmatch(A::AbstractMatrix{T}, V::AbstractMatrix, kr::AbstractRange, jr::AbstractRange) where {T}
    u, l = bandwidths(A)
    jj = 1
    for j in jr
        kk = 1
        for k in kr
            if !(-l ≤ j - k ≤ u) && V[kk, jj] ≠ zero(T)
                # we index V manually in column-major order
                throw(BandError(A, (k,j)))
            end
            kk += 1
        end
        jj += 1
    end
end

checkbandmatch(A::AbstractMatrix, V::AbstractMatrix, ::Colon, ::Colon) =
    checkbandmatch(A, V, 1:size(A,1), 1:size(A,2))

"""
    BandSlice(band::Band, indices::StepRange{Int,Int}) <: OrdinalRange{Int,Int}

Represent a `StepRange` of indices corresponding to a band.

Upon calling `to_indices()`, `Band`s are converted to `BandSlice` objects to represent
the indices over which the `Band` spans.

This mimics the relationship between `Colon` and `Base.Slice`.

# Example
```jldoctest
julia> B = BandedMatrix(0 => 1:4, 1=>1:3);

julia> to_indices(B, (Band(1),))[1]
BandSlice(Band(1), 5:5:15)
```
"""
struct BandSlice <: OrdinalRange{Int,Int}
    band::Band
    indices::StepRange{Int,Int}
end

for f in (:indices, :unsafe_indices, :axes1, :first, :last, :size, :length,
          :unsafe_length, :start, :step)
    @eval $f(S::BandSlice) = $f(S.indices)
end

@propagate_inbounds getindex(S::BandSlice, i::Union{Int, AbstractRange}) = getindex(S.indices, i)
show(io::IO, r::BandSlice) = print(io, "BandSlice(", r.band, ", ", r.indices, ")")

to_index(::Band) = throw(ArgumentError("Block must be converted by to_indices(...)"))

"""
" the following is designed to supported infinite baned arrays
"""

band_to_indices(A, _, b) = (BandSlice(b, diagind(A, b.i)),)
@inline to_indices(A, I::Tuple{Band}) = band_to_indices(A, axes(A), I[1])
view(A::AbstractArray, I::Band) = view(A, to_indices(A, (I,))...)
