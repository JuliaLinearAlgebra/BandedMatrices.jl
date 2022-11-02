module BandedMatrices
using Base, FillArrays, ArrayLayouts, LinearAlgebra, SparseArrays

using Base: require_one_based_indexing, reindex, checkbounds, @propagate_inbounds,
            oneto, promote_op, MultiplicativeInverses, OneTo, ReshapedArray, Slice
import Base: axes, axes1, getproperty, getindex, setindex!, *, +, -, ==, <, <=, >,
               >=, /, ^, \, adjoint, transpose, showerror, convert, size, view,
               unsafe_indices, first, last, size, length, unsafe_length, step, to_indices,
               to_index, show, fill!, similar, copy, promote_rule, IndexStyle, real, imag,
               copyto!

using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
import Base.Broadcast: BroadcastStyle, broadcasted, materialize, materialize!

using LinearAlgebra: AbstractTriangular, AdjOrTrans, BlasInt, BlasReal, BlasFloat, BlasComplex,
            checksquare, HermOrSym, chkstride1, QRPackedQ, StructuredMatrixStyle,
            checknonsingular, ipiv2perm, Givens
import LinearAlgebra: axpy!, _chol!, rot180, dot, cholcopy, _apply_ipiv_rows!,
            _apply_inverse_ipiv_rows!, diag, eigvals!, eigvals, eigen!, eigen,
            qr, qr!, ldiv!, mul!, lu, lu!, ldlt, ldlt!,
            kron, lmul!, rmul!, factorize, logabsdet,
            svdvals, svdvals!, tril!, triu!, diagzero

using LinearAlgebra.LAPACK
using LinearAlgebra.LAPACK: chkuplo, chktrans

import SparseArrays: sparse

import ArrayLayouts: MemoryLayout, transposelayout, triangulardata,
                    conjlayout, symmetriclayout, symmetricdata,
                    triangularlayout, MatLdivVec, hermitianlayout, hermitiandata,
                    materialize!, BlasMatMulMatAdd, BlasMatMulVecAdd, BlasMatLmulVec, BlasMatLdivVec,
                    colsupport, rowsupport, symmetricuplo, MatMulMatAdd, MatMulVecAdd,
                    sublayout, sub_materialize, _fill_lmul!, _copy_oftype,
                    reflector!, reflectorApply!, _copyto!, checkdimensions,
                    _qr!, _qr, _lu!, _lu, _factorize, AbstractTridiagonalLayout, TridiagonalLayout,
                    BidiagonalLayout, bidiagonaluplo, diagonaldata, supdiagonaldata, subdiagonaldata

import FillArrays: AbstractFill, getindex_value, _broadcasted_zeros, unique_value

const libblas = Base.libblas_name
const liblapack = Base.liblapack_name

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
include("generic/indexing.jl")


include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/bandedqr.jl")
include("banded/gbmm.jl")
include("banded/linalg.jl")
include("banded/dot.jl")

include("symbanded/symbanded.jl")
include("symbanded/ldlt.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/SplitCholesky.jl")
include("symbanded/bandedeigen.jl")
include("symbanded/tridiagonalize.jl")

include("tribanded.jl")

include("interfaceimpl.jl")


# function _precompile_()
#     precompile(Tuple{typeof(gbmm!), Char, Char, Float64, BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}}, BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}}, Float64, BandedMatrix{Float64,Array{Float64,2},Base.OneTo{Int64}}})
# end

# _precompile_()

end #module
