__precompile__()

module BandedMatrices
using Base, Compat

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror

import Base: convert, size, view

import Base.BLAS: libblas
import Base.LAPACK: liblapack


import Base.LinAlg: BlasInt,
                    BlasReal,
                    BlasFloat,
                    BlasComplex,
                    axpy!,
                    A_mul_B!,
                    A_ldiv_B!,
                    At_ldiv_B!,
                    Ac_ldiv_B!,
                    copy_oftype

import Base: lufact, cholfact, cholfact!

export BandedMatrix,
       SymBandedMatrix,
       bandrange,
       bzeros,
       beye,
       brand,
       bones,
       bandwidth,
       BandError,
       band,
       BandRange,
       bandwidths,
       colrange,
       rowrange



include("blas.jl")
include("lapack.jl")

include("AbstractBandedMatrix.jl")
include("Band.jl")
include("utils.jl")

include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/BandedQR.jl")
include("banded/gbmm.jl")
include("banded/linalg.jl")
include("banded/utils.jl")

include("symbanded/SymBandedMatrix.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/linalg.jl")

include("interface.jl")
include("deprecate.jl")


# include("precompile.jl")
# _precompile_()

end #module
