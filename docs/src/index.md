# BandedMatrices.jl Documentation


## Creating banded matrices

```@docs
BandedMatrix
```

```@docs
bones
```

```@docs
brand
```

```@meta
DocTestSetup = quote
    using BandedMatrices
end
```

To create a banded matrix of all zeros, identity matrix, or with a constant value
use the following constructors:
```jldoctest
julia> BandedMatrix(Zeros(5,5), (1,2))
5×5 BandedMatrix{Float64,Array{Float64,2}}:
 0.0  0.0  0.0   ⋅    ⋅
 0.0  0.0  0.0  0.0   ⋅
  ⋅   0.0  0.0  0.0  0.0
  ⋅    ⋅   0.0  0.0  0.0
  ⋅    ⋅    ⋅   0.0  0.0

julia> BandedMatrix(Eye(5), (1,2))
5×5 BandedMatrix{Float64,Array{Float64,2}}:
 1.0  0.0  0.0   ⋅    ⋅
 0.0  1.0  0.0  0.0   ⋅
  ⋅   0.0  1.0  0.0  0.0
  ⋅    ⋅   0.0  1.0  0.0
  ⋅    ⋅    ⋅   0.0  1.0

julia> BandedMatrix(Ones(5,5), (1,2))
5×5 BandedMatrix{Float64,Array{Float64,2}}:
 1.0  1.0  1.0   ⋅    ⋅
 1.0  1.0  1.0  1.0   ⋅
  ⋅   1.0  1.0  1.0  1.0
  ⋅    ⋅   1.0  1.0  1.0
  ⋅    ⋅    ⋅   1.0  1.0
```
To create a banded matrix of a given size with constant bands (such as the classical finite difference approximation of the one-dimensional Laplacian on the unit interval [0,1]), you can use the following:
```julia
n = 128
h = 1/n
A = BandedMatrix{Float64}(undef, (n,n), (1,1))
A[band(0)] .= -2/h^2
A[band(1)] .= A[band(-1)] .= 1/h^2
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

Use `Symmetric(::BandedMatrix)` to work with symmetric banded matrices.

## Banded matrix interface

Banded matrices go beyond the type `BandedMatrix`: one can also create
matrix types that conform to the _banded matrix interface_, in which case
many of the utility functions in this package are available. The banded matrix
interface consists of the following:

| Required methods | Brief description |
| :--------------- | :--------------- |
| `bandwidths(A)` | Returns a tuple containing the sub-diagonal and super-diagonal bandwidth |
| `isbanded(A)`    | Override to return `true` |
| :--------------- | :--------------- |
| Optional methods | Brief description |
| :--------------- | :--------------- |
| `inbands_getindex(A, k, j)` | Unsafe: return `A[k,j]`, without the need to check if we are inside the bands |
| `inbands_setindex!(A, v, k, j)` | Unsafe: set `A[k,j] = v`, without the need to check if we are inside the bands |
| `BandedMatrices.MemoryLayout(A)` | Override to get banded lazy linear algebra, e.g. `y .= Mul(A,x)` |
| `BandedMatrices.bandeddata(A)` | Override to return a matrix of the entries in BLAS format. Required if `MemoryLayout(A)` returns `BandedColumnMajor` |

Note that certain `SubArray`s of `BandedMatrix` are also banded matrices.
The banded matrix interface is implemented for such `SubArray`s to take advantage of this.


## Eigenvalues
To compute efficiently a selection of eigenvalues for a `BandedMatrix`, you may use any Krylov method that relies on a sequence of matrix * vector operations. For instance, using the package [KrylovKit](https://github.com/Jutho/KrylovKit.jl):
```julia
using KrylovKit
A = BandedMatrix(Eye(5), (1, 1))
KrylovKit.eigsolve(A, 1, :LR)
```


## Implementation

Currently, only column-major ordering is supported: a banded matrix `B`
```julia
[ a_11 a_12  ⋅    ⋅
  a_21 a_22 a_23  ⋅
  a_31 a_32 a_33 a_34
   ⋅   a_42 a_43 a_44  ]
```
is represented as a `BandedMatrix` with a field `B.data` representing the matrix as
```julia
[ ⋅     a_12   a_23    a_34
 a_11   a_22   a_33    a_44
 a_21   a_32   a_43    ⋅
 a_31   a_42   ⋅       ⋅       ]
```        
`B.l` gives the number of subdiagonals (2) and `B.u` gives the number of super-diagonals (1).  Both `B.l` and `B.u` must be non-negative at the moment.
