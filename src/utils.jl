# helper functions in blas routines
blas_size(t::Char, A::AbstractMatrix) = t == 'N' ? size(A) : (size(A, 2), size(A, 1))
blas_bandwidths(t::Char, A::AbstractMatrix) = t == 'N' ? bandwidths(A) : (bandwidth(A, 2), bandwidth(A, 1))
blas_view(t::Char, A::AbstractMatrix, I1, I2) = t == 'N' ? view(A, I1, I2) : view(A, I2, I1)

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


# check dimensions of inputs
checkdimensions(sizedest::Tuple{Int, Vararg{Int}}, sizesrc::Tuple{Int, Vararg{Int}}) =
    (sizedest == sizesrc ||
        throw(DimensionMismatch("tried to assign $(sizesrc) sized " *
                                "array to $(sizedest) destination")) )

checkdimensions(dest::AbstractVector, src::AbstractVector) =
    checkdimensions(size(dest), size(src))

checkdimensions(ldest::Int, src::AbstractVector) =
    checkdimensions((ldest, ), size(src))

checkdimensions(kr::Range, jr::Range, src::AbstractMatrix) =
    checkdimensions((length(kr), length(jr)), size(src))


# return the bandwidths of A*B
function prodbandwidths(A::AbstractMatrix, B::AbstractMatrix)
    m = size(A, 1)
    n = size(B, 2)
    bandwidth(A, 1) + bandwidth(B, 1), bandwidth(A, 2) + bandwidth(B, 2)
end

# return the bandwidths of A+B
function sumbandwidths(A::AbstractMatrix, B::AbstractMatrix)
    max(bandwidth(A, 1), bandwidth(B, 1)), max(bandwidth(A, 2), bandwidth(B, 2))
end
