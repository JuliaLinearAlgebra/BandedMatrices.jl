# BandedMatrices.jl
A Julia package for representing banded matrices

[![Build Status](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/BandedMatrices.jl) 
[![Build status](https://ci.appveyor.com/api/projects/status/1nbjit44621pa5e5?svg=true)](https://ci.appveyor.com/project/dlfivefifty/bandedmatrices-jl)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMatrices.github.io/BandedMatrices.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMatrices.github.io/BandedMatrices.jl/latest)
[![codecov](https://codecov.io/gh/JuliaMatrices/BandedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/BandedMatrices.jl)



This package supports representing banded matrices by only the entries on the
bands.  

One can create banded matrices of type `BandedMatrix` as follows:

```julia
BandedMatrix(Zeros(m,n), (l,u))    # creates a banded matrix of zeros, with l sub-diagonals and u super-diagonals
brand(m,n,l,u)     # creates a random banded matrix, with l sub-diagonals and u super-diagonals
BandedMatrix(Ones(m,n), (l,u))     # creates a banded matrix of ones, with l sub-diagonals and u super-diagonals
BandedMatrix(Eye(n), (l,u))        # creates a banded  n x n identity matrix, with l sub-diagonals and u super-diagonals
BandedMatrix(-1=> 1:5, 2=>1:3)     # creates a 5 x 5 banded matrix version of diagm(-1=> 1:5, 2=>1:3)
BandedMatrix((-1=> 1:5, 2=>1:3), (n,m))     # creates an n x m banded matrix with 1 sub-diagonals and u super-diagonals with the specified diagonals
BandedMatrix((-1=> 1:5, 2=>1:3), (n,m), (l,u))     # creates an n x m banded matrix with l sub-diagonals and u super-diagonals with the specified diagonals
```
For more examples, see [the documentation](http://juliamatrices.github.io/BandedMatrices.jl/latest/#Creating-banded-matrices-1).

Specialized algebra routines are overriden, include `*` and `\`:

```julia
A = brand(10000,10000,4,3)  # creates a 10000 x 10000 matrix with 4 sub-diagonals
                            # and 3 super-diagonals
b = randn(10000)
A*b  #   Calls optimized matrix*vector routine
A*A  #   Calls optimized matrix*matrix routine
A\b  #   Calls optimized matrix\vector routine
```
