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

# BLAS/linear algebra overrides

@inline dot(x...) = Base.dot(x...)
@inline dot{T<:Union{Float64,Float32}}(M::Int,a::Ptr{T},incx::Int,b::Ptr{T},incy::Int) =
    BLAS.dot(M,a,incx,b,incy)
@inline dot{T<:Union{Complex128,Complex64}}(M::Int,a::Ptr{T},incx::Int,b::Ptr{T},incy::Int) =
    BLAS.dotc(M,a,incx,b,incy)

dotu{T<:Union{Complex64,Complex128}}(f::StridedVector{T},g::StridedVector{T}) =
    BLAS.dotu(f,g)
dotu{N<:Real}(f::AbstractVector{Complex{Float64}},g::AbstractVector{N}) = dot(conj(f),g)
dotu{N<:Real,T<:Number}(f::AbstractVector{N},g::AbstractVector{T}) = dot(f,g)


normalize!(w::AbstractVector) = scale!(w,inv(norm(w)))
normalize!{T<:BlasFloat}(w::Vector{T}) = normalize!(length(w),w)
normalize!{T<:Union{Float64,Float32}}(n,w::Union{Vector{T},Ptr{T}}) =
    BLAS.scal!(n,inv(BLAS.nrm2(n,w,1)),w,1)
normalize!{T<:Union{Complex128,Complex64}}(n,w::Union{Vector{T},Ptr{T}}) =
    BLAS.scal!(n,T(inv(BLAS.nrm2(n,w,1))),w,1)



# AbstractBandedMatrix must implement

@compat abstract type AbstractBandedMatrix{T} <: AbstractSparseMatrix{T,Int} end

doc"""
    bandwidths(A)

Returns a tuple containing the upper and lower bandwidth of `A`.
"""
bandwidths(A::AbstractArray) = bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractArray) = -bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractArray,k::Integer) = k==1 ? -bandwidth(A,1) : bandwidth(A,2)


doc"""
    bandwidth(A,i)

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).
"""
bandwidth(A::DenseVecOrMat,k::Integer) = k==1 ? size(A,1)-1 : size(A,2)-1
if isdefined(Base, :RowVector)
    bandwidth{T,DV<:DenseVector}(A::RowVector{T,DV},k::Integer) = k==1 ? size(A,1)-1 : size(A,2)-1
end

doc"""
    bandrange(A)

Returns the range `-bandwidth(A,1):bandwidth(A,2)`.
"""
bandrange(A::AbstractBandedMatrix) = -bandwidth(A,1):bandwidth(A,2)



doc"""
    isbanded(A)

returns true if a matrix implements the banded interface.
"""
isbanded(::AbstractBandedMatrix) = true
isbanded(::) = false

# override bandwidth(A,k) for each AbstractBandedMatrix
# override inbands_getindex(A,k,j)


include("blas.jl")
include("lapack.jl")

include("banded/BandedMatrix.jl")
include("banded/BandedLU.jl")
include("banded/BandedQR.jl")
include("banded/linalg.jl")

include("symbanded/SymBandedMatrix.jl")
include("symbanded/BandedCholesky.jl")
include("symbanded/linalg.jl")


end #module
