module BandedMatrices
using Base, FillArrays, ArrayLayouts, LinearAlgebra

using Base: require_one_based_indexing, reindex, checkbounds, @propagate_inbounds,
            oneto, promote_op, OneTo, ReshapedArray, Slice
import Base: axes, axes1, getproperty, getindex, setindex!, *, +, -, ==, <, <=, >,
               >=, /, \, adjoint, transpose, showerror, convert, size, view,
               unsafe_indices, first, last, size, length, unsafe_length, step, to_indices,
               to_index, show, fill!, similar, copy, promote_rule, real, imag,
               copyto!, Array

using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Broadcasted
import Base.Broadcast: BroadcastStyle, broadcasted

import Base: kron, rot180

import LinearAlgebra: _apply_inverse_ipiv_rows!, _apply_ipiv_rows!, _chol!, axpy!, cholcopy, diag, diagzero, dot, eigen,
                      eigen!, eigvals, eigvals!, factorize, isdiag, istril, istriu, ldiv!, ldlt, ldlt!, lmul!,
                      logabsdet, lu, lu!, mul!, qr, qr!, rmul!, svdvals, svdvals!, tril!, triu!

using LinearAlgebra: AbstractTriangular, AdjOrTrans, BlasComplex, BlasFloat, BlasInt, BlasReal, Givens, HermOrSym,
                     QRPackedQ, RealHermSymComplexHerm, StructuredMatrixStyle, checksquare, chkstride1, ipiv2perm

using LinearAlgebra.LAPACK

using LinearAlgebra.LAPACK: chktrans, chkuplo

import ArrayLayouts: AbstractTridiagonalLayout, BidiagonalLayout, BlasMatLdivVec, BlasMatLmulVec,
                     BlasMatMulMatAdd, BlasMatMulVecAdd, MatMulMatAdd, MatMulVecAdd, MemoryLayout,
                     _copy_oftype, _copyto!, _factorize, _lu, _lu!, _qr, _qr!, bidiagonaluplo, checkdimensions,
                     colsupport, conjlayout, copymutable_oftype_layout, diagonaldata, dualadjoint, hermitiandata,
                     hermitianlayout, materialize, materialize!, reflector!, reflectorApply!, rowsupport,
                     sub_materialize, subdiagonaldata, sublayout, supdiagonaldata, symmetricdata, symmetriclayout,
                     symmetricuplo, transposelayout, triangulardata, triangularlayout, zero!

import FillArrays: AbstractFill, getindex_value, _broadcasted_zeros, unique_value, OneElement, RectDiagonal

const libblas = LinearAlgebra.BLAS.libblas
const liblapack = LinearAlgebra.BLAS.liblapack
const AdjointFact = isdefined(LinearAlgebra, :AdjointFactorization) ?
    LinearAlgebra.AdjointFactorization :
    Adjoint
const TransposeFact = isdefined(LinearAlgebra, :TransposeFactorization) ?
    LinearAlgebra.TransposeFactorization :
    Transpose

export BandedMatrix,
       bandrange,
       brand,
       brandn,
       bandwidth,
       BandError,
       band,
       Band,
       BandRange,
       bandwidths,
       colrange,
       rowrange,
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

if !isdefined(Base, :get_extension)
    include("../ext/BandedMatricesSparseArraysExt.jl")
end

include("precompile.jl")


end #module
