module BandedMatrices
using Base, FillArrays, LazyArrays

using LinearAlgebra, SparseArrays, Random
using LinearAlgebra.LAPACK
import Base: axes, axes1, getproperty, iterate
import LinearAlgebra: BlasInt,
                    BlasReal,
                    BlasFloat,
                    BlasComplex,
                    axpy!,
                    copy_oftype,
                    checksquare,
                    adjoint,
                    transpose,
                    AdjOrTrans, HermOrSym
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, norm, diag, eigvals!, eigvals,
            qr, axpy!, ldiv!, mul!, lu, lu!, AbstractTriangular, has_offset_axes,
            chkstride1, kron, lmul!, rmul!
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

import LazyArrays: MemoryLayout, blasmul!, @blasmatvec, @blasmatmat, @lazymul, @lazylmul, @lazyldiv,
                    AbstractStridedLayout, AbstractColumnMajor, AbstractRowMajor,
                    _copyto!, MatMulVec, MatMulMat, transposelayout, triangulardata,
                    ConjLayout, conjlayout, SymmetricLayout, symmetriclayout, symmetricdata,
                    triangularlayout, InverseLayout, MatMulVec, MatLdivVec, TriangularLayout,
                    ArrayMulArrayStyle, HermitianLayout, hermitianlayout, hermitiandata,
                    BLASMul, materialize!
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


include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/BandedQR.jl")
include("banded/gbmm.jl")
include("banded/linalg.jl")

include("symbanded/symbanded.jl")
include("symbanded/BandedCholesky.jl")

include("tribanded.jl")

include("interfaceimpl.jl")

@deprecate setindex!(A::BandedMatrix, v, b::Band) A[b] .= v
@deprecate setindex!(A::BandedMatrix, v, ::BandRangeType, j::Integer) A[BandRange,j] .= v
@deprecate setindex!(A::BandedMatrix, v, kr::Colon, j::Integer) A[:,j] .= v
@deprecate setindex!(A::BandedMatrix, v, kr::AbstractRange, j::Integer) A[kr,j] .= v
@deprecate setindex!(A::BandedMatrix, v, ::Colon, ::Colon) A[:,:] .= v
@deprecate setindex!(A::BandedMatrix, v, ::Colon) A[:] .= v
@deprecate setindex!(A::BandedMatrix, v, k::Integer, ::BandRangeType) A[k,BandRange] .= v
@deprecate setindex!(A::BandedMatrix, v, k::Integer, jr::AbstractRange) A[k,jr] .= v
@deprecate setindex!(A::BandedMatrix, v, kr::AbstractRange, jr::AbstractRange) A[kr,jr] .= v
@deprecate setindex!(A::BandedMatrix, v, ::BandRangeType) BandedMatrices.bandeddata(A) .= v

end #module
