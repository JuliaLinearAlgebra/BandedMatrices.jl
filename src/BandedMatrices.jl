__precompile__()

module BandedMatrices
using Base, FillArrays

using LinearAlgebra, SparseArrays, Random
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
            qr, axpy!, ldiv!, mul!, lu, lu!
import SparseArrays: sparse

import Base: getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer, copyto!

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted

import LazyArrays: MemoryLayout, blasmul!, @blasmatvec, @blasmatmat

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
include("memorylayout.jl")

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
