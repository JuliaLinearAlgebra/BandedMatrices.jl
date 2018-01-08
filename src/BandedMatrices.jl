__precompile__()

module BandedMatrices
using Base, Compat, FillArrays

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, indices, unsafe_indices, indices1,
                first, last, size, length, unsafe_length, start, next, done, step,
                to_indices, to_index, indices, show, fill!, copy!

import Base.BLAS: libblas
import Base.LAPACK: liblapack


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

import Base: lufact, cholfact, cholfact!, promote_op

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
