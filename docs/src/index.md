# BandedMatrices.jl Documentation


## Creating banded matrices

```@docs
BandedMatrix
```

```@docs
beye
```

```@docs
bones
```

```@docs
brand
```

```@docs
bzeros
```


## Accessing banded matrices

```@docs
bandwidths
```

```@docs
bandwidth
```

```@docs
bandrange
```

```@docs
band
```

```@docs
BandRange
```



## Creating symmetric banded matrices

```@docs
SymBandedMatrix
```

```@docs
sbeye
```

```@docs
sbones
```

```@docs
sbrand
```

```@docs
sbzeros
```


## Banded matrix interface

Banded matrices go beyond the type `BandedMatrix`: one can also create
matrix types that conform to the _banded matrix interface_, in which case
many of the utility functions in this package are available. The banded matrix
interface consists of the following:

| Required methods | Brief description |
| --------------- | --------------- |
| `bandwidth(A, k)` | Returns the sub-diagonal bandwidth (`k==1`) or the super-diagonal bandwidth (`k==2`) |
| `isbanded(A)`    | Override to return `true` |
| `inbands_getindex(A, k, j)` | Unsafe: return `A[k,j]`, without the need to check if we are inside the bands |
| `inbands_setindex!(A, v, k, j)` | Unsafe: set `A[k,j] = v`, without the need to check if we are inside the bands |

Note that certain `SubArray`s of `BandedMatrix` are also banded matrices.
The banded matrix interface is implemented for such `SubArray`s to take advantage of this.
