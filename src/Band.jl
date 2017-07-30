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
    A::AbstractMatrix
    i::Int
end

function showerror(io::IO, e::BandError)
    A, i = e.A, e.i
    print(io, "attempt to access $(typeof(A)) with bandwidths " *
              "($(bandwidth(A, 1)), $(bandwidth(A, 2))) at band $i")
end


# length of diagonal
@inline diaglength(A::AbstractMatrix, b::Band) = diaglength(A, b.i)
@inline function diaglength(A::AbstractMatrix, i::Integer)
    max(min(size(A, 2), size(A, 1)+i) - max(0, i), 0)
end


# check indices fall in the band
checkband(A::AbstractMatrix, i::Integer) =
    (bandinds(A, 1) ≤ i ≤ bandinds(A, 2) || throw(BandError(A, i)))

checkband(A::AbstractMatrix, b::Band) = checkband(A, b.i)

checkband(A::AbstractMatrix, k::Integer, j::Integer) = checkband(A, j-k)

checkband(A::AbstractMatrix, kr::Range, j::Integer) =
    (checkband(A, first(kr), j); checkband(A,  last(kr), j))

checkband(A::AbstractMatrix, k::Integer, jr::Range) =
    (checkband(A, k, first(jr)); checkband(A, k,  last(jr)))

checkband(A::AbstractMatrix, kr::Range, jr::Range) =
    (checkband(A, kr, first(jr)); checkband(A, kr,  last(jr)))


# checks if the bands match A
function checkbandmatch{T}(A::AbstractMatrix{T}, V::AbstractVector, ::Colon, j::Integer)
    for k = 1:colstart(A,j)-1
        if V[k] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
    for k = colstop(A,j)+1:size(A,1)
        if V[k] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::AbstractMatrix{T}, V::AbstractVector, kr::Range, j::Integer)
    a = colstart(A, j)
    b = colstop(A, j)
    i = 0
    for v in V
        k = kr[i+=1]
        if (k < a || k > b) && v ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::AbstractMatrix{T}, V::AbstractVector, k::Integer, ::Colon)
    for j = 1:rowstart(A,k)-1
        if V[j] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
    for j = rowstop(A,j)+1:size(A,2)
        if V[j] ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::AbstractMatrix{T}, V::AbstractVector, k::Integer, jr::Range)
    a = rowstart(A, k)
    b = rowstop(A, k)
    i = 0
    for v in V
        j = jr[i+=1]
        if (j < a || j > b) && v ≠ zero(T)
            throw(BandError(A, j-k))
        end
    end
end

function checkbandmatch{T}(A::AbstractMatrix{T}, V::AbstractMatrix, kr::Range, jr::Range)
    u, l = A.u, A.l
    jj = 1
    for j in jr
        kk = 1
        for k in kr
            if !(-l ≤ j - k ≤ u) && V[kk, jj] ≠ zero(T)
                # we index V manually in column-major order
                throw(BandError(A, j-k))
            end
            kk += 1
        end
        jj += 1
    end
end

checkbandmatch(A::AbstractMatrix, V::AbstractMatrix, ::Colon, ::Colon) =
    checkbandmatch(A, V, 1:size(A,1), 1:size(A,2))
