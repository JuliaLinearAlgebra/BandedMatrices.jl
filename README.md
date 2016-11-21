# BandedMatrices.jl
A Julia package for representing banded matrices

[![Build Status](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMatrices.github.io/BandedMatrices.jl/stable) 
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMatrices.github.io/BandedMatrices.jl/latest)



This package supports representing banded matrices by only the entries on the
bands.  

One can create banded matrices of type `BandedMatrix` as follows:

```julia
bzeros(m,n,l,u)    # creates a banded matrix of zeros, with l sub-diagonals and u super-diagonals
brand(m,n,l,u)     # creates a random banded matrix, with l sub-diagonals and u super-diagonals
bones(m,n,l,u)     # creates a banded matrix of ones, with l sub-diagonals and u super-diagonals
beye(n,l,u)        # creates a banded  n x n identity matrix, with l sub-diagonals and u super-diagonals
```

Specialized algebra routines are overriden, include `*` and `\`:

```julia
A = brand(10000,10000,4,3)  # creates a 10000 x 10000 matrix with 4 sub-diagonals
                            # and 3 super-diagonals
b = randn(10000)
A*b  #   Calls optimized matrix*vector routine
A*A  #   Calls optimized matrix*matrix routine
A\b  #   Calls optimized matrix\vector routine
```


## Implementation

Currently, only column-major ordering is supported: a banded matrix `B`
```julia
[ a_11 a_12
  a_21 a_22 a_23
  a_31 a_32 a_33 a_34
       a_42 a_43 a_44  ]
```
is represented as a `BandedMatrix` with a field `B.data` representing the matrix as
```julia
[ *     a_12   a_23    a_34
 a_11   a_22   a_33    a_43
 a_21   a_32   a_43    *
 a_31   a_42   *       *       ]
```        
`B.l` gives the number of subdiagonals (2) and `B.u` gives the number of super-diagonals (1).  Both `B.l` and `B.u` must be non-negative at the moment.
