__precompile__()

module BandedMatrices
using Base, Compat, FillArrays
if VERSION ≥ v"0.7-"
    using LinearAlgebra, SparseArrays, Compat.Random
    using LinearAlgebra.LAPACK
    import Base: axes1, getproperty, iterate
    import LinearAlgebra: BlasInt,
                        BlasReal,
                        BlasFloat,
                        BlasComplex,
                        axpy!,
                        copy_oftype,
                        checksquare,
                        adjoint,
                        transpose
   import LinearAlgebra.BLAS: libblas
   import LinearAlgebra.LAPACK: liblapack
   import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals,
                qr, axpy!, ldiv!, mul!
   import SparseArrays: sparse

   const lufact = LinearAlgebra.lu # TODO: Remove once 0.6 is dropped
else
    import Base.LinAlg: BlasInt,
                        BlasReal,
                        BlasFloat,
                        BlasComplex,
                        axpy!,
                        A_mul_B!,
                        Ac_mul_B,
                        Ac_mul_B!,
                        A_mul_Bc,
                        A_mul_Bc!,
                        Ac_mul_Bc,
                        Ac_mul_Bc!,
                        At_mul_B,
                        At_mul_B!,
                        A_mul_Bt,
                        A_mul_Bt!,
                        At_mul_Bt,
                        At_mul_Bt!,
                        A_ldiv_B!,
                        At_ldiv_B!,
                        Ac_ldiv_B!,
                        copy_oftype,
                        checksquare
   using Base.LAPACK
   import Base.BLAS: libblas
   import Base.LAPACK: liblapack
   import Base: lufact, cholfact, cholfact!, norm, diag, eigvals!, eigvals,
                At_mul_B, Ac_mul_B, A_mul_B!, qr, qrfact, axpy!
   import Base: sparse, indices1, indices

   rmul!(A::AbstractArray, b::Number) = scale!(A, b)
   lmul!(a::Number, B::AbstractArray) = scale!(a, B)
   parentindices(A) = parentindexes(A)
   const axes1 = indices1
   const adjoint = Base.ctranspose
   const cholesky = Base.cholfact
   const cholesky! = Base.cholfact!
end

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, copy!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer


import FillArrays: AbstractFill

export BandedMatrix,
       SymBandedMatrix,
       bandrange,
       brand,
       bandwidth,
       BandError,
       band,
       Band,
       BandRange,
       bandwidths,
       colrange,
       rowrange,
       isbanded,
       Zeros,
       Fill,
       Ones,
       Eye



include("memorylayout.jl")
include("blas.jl")
include("lapack.jl")

include("generic/AbstractBandedMatrix.jl")
include("generic/Band.jl")
include("generic/utils.jl")
include("generic/interface.jl")

include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/BandedQR.jl")
include("banded/gbmm.jl")
include("banded/linalg.jl")

include("symbanded/SymBandedMatrix.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/linalg.jl")

include("interfaceimpl.jl")

include("deprecate.jl")


# include("precompile.jl")
# _precompile_()

end #module
