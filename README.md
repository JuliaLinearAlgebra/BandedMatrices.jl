# BandedMatrices.jl
A Julia package for representing banded matrices

[![Build Status](https://travis-ci.org/ApproxFun/BandedMatrices.jl.svg?branch=master)](https://travis-ci.org/ApproxFun/BandedMatrices.jl)



This package supports representing banded matrices by only the entries on the
bands.  Currently, only column-major ordering is supported: a banded matrix
```julia
 [ a_11 a_12
   a_21 a_22 a_23
   a_31 a_32 a_33 a_34
        a_42 a_43 a_44  ]
```
is represented by the matrix       
```julia
[ *      a_12   a_23    a_34
  a_11   a_22   a_33    a_43
  a_21   a_32   a_43    *
  a_31   a_42   *       *       ]
```        

One can create banded matrices as follows:

```julia
bzeros(m,n,l,u)    # creates a banded matrix of zeros, with l subdiagonals and u superdiagonals
brand(m,n,l,u)     # creates a random banded matrix, with l subdiagonals and u superdiagonals
bones(m,n,l,u)     # creates a banded matrix of ones, with l subdiagonals and u superdiagonals
beye(n,l,u)        # creates a banded  n x n identity matrix, with l subdiagonals and u superdiagonals
```
