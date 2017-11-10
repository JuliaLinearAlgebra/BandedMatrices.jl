__precompile__()

module BandedMatrices
using Base, Compat

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds

import Base: convert, size, view

import Base.BLAS: libblas
import Base.LAPACK: liblapack,
                    chklapackerror


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
                    checksquare,
                    Eigen

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

include("symbanded/SymBandedMatrix.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/linalg.jl")

include("interface.jl")
include("deprecate.jl")


# include("precompile.jl")
# _precompile_()

end #module
