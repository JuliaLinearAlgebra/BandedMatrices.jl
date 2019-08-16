module BandedMatrices
using Base, FillArrays, LazyArrays, MatrixFactorizations, LinearAlgebra, SparseArrays, Random
using LinearAlgebra.LAPACK
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
                        copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym,
                        _chol!
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, cholcopy, norm, diag, eigvals!, eigvals, eigen!, eigen,
            qr, qr!, axpy!, ldiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular,
            chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, logabsdet,
            svdvals, svdvals!, QRPackedQ, checknonsingular, ipiv2perm, _apply_ipiv!, tril!,
            triu!, Givens
import MatrixFactorizations: ql, ql!, QLPackedQ, reflector!, reflectorApply!
import SparseArrays: sparse

import Base: getindex, setindex!, *, +, -, ==, <, <=, >, isassigned,
                >=, /, ^, \, transpose, showerror, reindex, checkbounds, @propagate_inbounds

import Base: convert, size, view, unsafe_indices,
                first, last, size, length, unsafe_length, step,
                to_indices, to_index, show, fill!, promote_op,
                MultiplicativeInverses, OneTo, ReshapedArray,
                               similar, copy, convert, promote_rule, rand,
                            IndexStyle, real, imag, Slice, pointer, unsafe_convert, copyto!,
                            hcat, vcat, hvcat

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted,
                        materialize, materialize!

import LazyArrays: MemoryLayout, @lazymul, @lazylmul, @lazyldiv,
                    AbstractStridedLayout, AbstractColumnMajor, AbstractRowMajor,
                    transposelayout, triangulardata,
                    ConjLayout, conjlayout, SymmetricLayout, symmetriclayout, symmetricdata,
                    triangularlayout, MatLdivVec, TriangularLayout,
                    AbstractBandedLayout, DiagonalLayout,
                    HermitianLayout, hermitianlayout, hermitiandata,
                    MulAdd, materialize!, BlasMatMulMatAdd, BlasMatMulVecAdd, BlasMatLmulVec, BlasMatLdivVec,
                    VcatLayout, ZerosLayout,
                    AbstractColumnMajor, MulLayout, colsupport, rowsupport,
                    DenseColumnMajor, DenseRowMajor, ApplyArrayBroadcastStyle,
                    mulapplystyle, AbstractMulAddStyle, symmetricuplo, MatMulMatAdd, MatMulVecAdd,
                    _fill_lmul!, applybroadcaststyle, subarraylayout, sub_materialize, lazy_getindex

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


if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

include("blas.jl")
include("lapack.jl")

include("generic/AbstractBandedMatrix.jl")
include("generic/broadcast.jl")
include("generic/matmul.jl")
include("generic/Band.jl")
include("generic/utils.jl")
include("generic/indexing.jl")


include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/bandedqr.jl")
include("banded/bandedql.jl")
include("banded/gbmm.jl")
include("banded/linalg.jl")

include("symbanded/symbanded.jl")
include("symbanded/ldlt.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/SplitCholesky.jl")
include("symbanded/bandedeigen.jl")

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
