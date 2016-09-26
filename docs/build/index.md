
<a id='BandedMatrices.jl-Documentation-1'></a>

# BandedMatrices.jl Documentation


<a id='Creating-banded-matrices-1'></a>

## Creating banded matrices

<a id='BandedMatrices.BandedMatrix' href='#BandedMatrices.BandedMatrix'>#</a>
**`BandedMatrices.BandedMatrix`** &mdash; *Type*.



```
BandedMatrix(T, n, m, l, u)
```

returns an unitialized `n`×`m` banded matrix of type `T` with bandwidths `(l,u)`.

<a id='BandedMatrices.beye' href='#BandedMatrices.beye'>#</a>
**`BandedMatrices.beye`** &mdash; *Function*.



```
beye(T,n,l,u)
```

`n×n` banded identity matrix of type `T` with bandwidths `(l,u)`

<a id='BandedMatrices.bones' href='#BandedMatrices.bones'>#</a>
**`BandedMatrices.bones`** &mdash; *Function*.



```
bones(T,n,m,l,u)
```

Creates an `n×m` banded matrix  with ones in the bandwidth of type `T` with bandwidths `(l,u)`

<a id='BandedMatrices.brand' href='#BandedMatrices.brand'>#</a>
**`BandedMatrices.brand`** &mdash; *Function*.



```
brand(T,n,m,l,u)
```

Creates an `n×m` banded matrix  with random numbers in the bandwidth of type `T` with bandwidths `(l,u)`

<a id='BandedMatrices.bzeros' href='#BandedMatrices.bzeros'>#</a>
**`BandedMatrices.bzeros`** &mdash; *Function*.



```
bzeros(T,n,m,l,u)
```

Creates an `n×m` banded matrix  of all zeros of type `T` with bandwidths `(l,u)`


<a id='Accessing-banded-matrices-1'></a>

## Accessing banded matrices

<a id='BandedMatrices.bandwidths' href='#BandedMatrices.bandwidths'>#</a>
**`BandedMatrices.bandwidths`** &mdash; *Function*.



```
bandwidths(A)
```

Returns a tuple containing the upper and lower bandwidth of `A`.

<a id='BandedMatrices.bandwidth' href='#BandedMatrices.bandwidth'>#</a>
**`BandedMatrices.bandwidth`** &mdash; *Function*.



```
bandwidth(A,i)
```

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).

<a id='BandedMatrices.bandrange' href='#BandedMatrices.bandrange'>#</a>
**`BandedMatrices.bandrange`** &mdash; *Function*.



```
bandrange(A)
```

Returns the range `-bandwidth(A,1):bandwidth(A,2)`.

<a id='BandedMatrices.band' href='#BandedMatrices.band'>#</a>
**`BandedMatrices.band`** &mdash; *Function*.



```
band(i)
```

Represents the `i`-th band of a banded matrix.

```jlcon
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

<a id='BandedMatrices.BandRange' href='#BandedMatrices.BandRange'>#</a>
**`BandedMatrices.BandRange`** &mdash; *Type*.



```
BandRange
```

Represents the entries in a row/column inside the bands.

```jlcon
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

