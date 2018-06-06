__precompile__()

module BandedMatrices
using Base, Compat, FillArrays
if VERSION ≥ v"0.7-"
    using LinearAlgebra, SparseArrays, Random
    using LinearAlgebra.LAPACK
    import LinearAlgebra: BlasInt,
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
                        copy_oftype,
                        checksquare,
                        adjoint,
                        transpose
   import LinearAlgebra.BLAS: libblas
   import LinearAlgebra.LAPACK: liblapack
   import LinearAlgebra: cholfact, cholfact!, norm, diag, eigvals!, eigvals,
                qr, qrfact, axpy!, ldiv!, mul!
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
   import Base: sparse

   rmul!(A::AbstractArray, b::Number) = scale!(A, b)
   lmul!(a::Number, B::AbstractArray) = scale!(a, B)
   parentindices(A) = parentindexes(A)
end

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, indices, unsafe_indices, indices1,
                first, last, size, length, unsafe_length, start, next, done, step,
                to_indices, to_index, indices, show, fill!, copy!, promote_op





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
