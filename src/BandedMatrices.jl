module BandedMatrices
using Base, FillArrays, LazyArrays

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
                    transpose,
                    AdjOrTrans
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
                            IndexStyle, real, imag, Slice, pointer, unsafe_convert, copyto!

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted

import LazyArrays: MemoryLayout, blasmul!, @blasmatvec, @blasmatmat, @lazymul,
                    AbstractStridedLayout, AbstractColumnMajor,
                    _copyto!, BMatVec, BMixedMatVec, BMixedMatMat, transposelayout,
                    ConjLayout, conjlayout, SymmetricLayout, symmetriclayout
import FillArrays: AbstractFill

export BandedMatrix,
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
include("generic/broadcast.jl")
include("generic/matmul.jl")
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

end #module
