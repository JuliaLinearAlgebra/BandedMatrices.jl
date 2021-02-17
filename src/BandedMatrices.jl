module BandedMatrices
using Base, FillArrays, ArrayLayouts, LinearAlgebra, SparseArrays, Random
using LinearAlgebra.LAPACK
import Base: axes, axes1, getproperty, iterate, tail
import LinearAlgebra: BlasInt, BlasReal, BlasFloat, BlasComplex, axpy!,
                        copy_oftype, checksquare, adjoint, transpose, AdjOrTrans, HermOrSym,
                        _chol!, rot180, dot
import LinearAlgebra.BLAS: libblas
import LinearAlgebra.LAPACK: liblapack, chkuplo, chktrans
import LinearAlgebra: cholesky, cholesky!, cholcopy, norm, diag, eigvals!, eigvals, eigen!, eigen,
            qr, qr!, axpy!, ldiv!, mul!, lu, lu!, ldlt, ldlt!, AbstractTriangular,
            chkstride1, kron, lmul!, rmul!, factorize, StructuredMatrixStyle, logabsdet,
            svdvals, svdvals!, QRPackedQ, checknonsingular, ipiv2perm, tril!,
            triu!, Givens, diagzero
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

import ArrayLayouts: MemoryLayout, transposelayout, triangulardata,
                    conjlayout, symmetriclayout, symmetricdata,
                    triangularlayout, MatLdivVec, hermitianlayout, hermitiandata,
                    materialize!, BlasMatMulMatAdd, BlasMatMulVecAdd, BlasMatLmulVec, BlasMatLdivVec,
                    colsupport, rowsupport, symmetricuplo, MatMulMatAdd, MatMulVecAdd,
                    sublayout, sub_materialize, _fill_lmul!,
                    reflector!, reflectorApply!, _copyto!, checkdimensions,
                    _qr!, _qr, _lu!, _lu, _factorize, AbstractTridiagonalLayout, TridiagonalLayout, BidiagonalLayout

import FillArrays: AbstractFill, getindex_value, _broadcasted_zeros, unique_value


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


import Base: require_one_based_indexing
import LinearAlgebra: _apply_ipiv_rows!

if VERSION < v"1.6-"
	oneto(n) = OneTo(n)
else
	import Base: oneto
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
