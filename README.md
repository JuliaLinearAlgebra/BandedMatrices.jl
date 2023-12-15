# BandedMatrices.jl
A Julia package for representing banded matrices

[![Build Status](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl/workflows/CI/badge.svg)](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaLinearAlgebra.github.io/BandedMatrices.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaLinearAlgebra.github.io/BandedMatrices.jl/dev)
[![codecov](https://codecov.io/gh/JuliaLinearAlgebra/BandedMatrices.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaLinearAlgebra/BandedMatrices.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![deps](https://juliahub.com/docs/General/BandedMatrices/stable/deps.svg)](https://juliahub.com/ui/Packages/General/BandedMatrices?t=2)
[![version](https://juliahub.com/docs/General/BandedMatrices/stable/version.svg)](https://juliahub.com/ui/Packages/General/BandedMatrices)
[![pkgeval](https://juliahub.com/docs/General/BandedMatrices/stable/pkgeval.svg)](https://juliahub.com/ui/Packages/General/BandedMatrices)


This package supports representing banded matrices by only the entries on the
bands.

One can create banded matrices of type `BandedMatrix` as follows:

```julia
BandedMatrix(-1=> 1:5, 2=>1:3)     # creates a 6 x 6 banded matrix version of diagm(-1=> 1:5, 2=>1:3)
BandedMatrix((-1=> 1:5, 2=>1:3), (n,m))     # creates an n x m banded matrix with 1 sub-diagonals and u super-diagonals with the specified diagonals
BandedMatrix((-1=> 1:5, 2=>1:3), (n,m), (l,u))     # creates an n x m banded matrix with l sub-diagonals and u super-diagonals with the specified diagonals
BandedMatrix(FillArrays.Zeros(m,n), (l,u))    # creates a banded matrix of zeros, with l sub-diagonals and u super-diagonals
BandedMatrix(FillArrays.Ones(m,n), (l,u))     # creates a banded matrix of ones, with l sub-diagonals and u super-diagonals
BandedMatrix(FillArrays.Eye(n), (l,u))        # creates a banded  n x n identity matrix, with l sub-diagonals and u super-diagonals
brand(m,n,l,u)     # creates a random banded matrix, with l sub-diagonals and u super-diagonals
```
For more examples, see [the documentation](https://julialinearalgebra.github.io/BandedMatrices.jl/dev/#Creating-banded-matrices).

Specialized algebra routines are overriden, include `*` and `\`:

```julia
A = brand(10000,10000,4,3)  # creates a 10000 x 10000 matrix with 4 sub-diagonals
                            # and 3 super-diagonals
b = randn(10000)
A*b  #   Calls optimized matrix*vector routine
A*A  #   Calls optimized matrix*matrix routine
A\b  #   Calls optimized matrix\vector routine
```
