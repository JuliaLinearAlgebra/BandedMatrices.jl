#
# Represent a symmetric banded matrix
# [ a_11 a_12 a_13
#   a_12 a_22 a_23 a_24
#   a_13 a_23 a_33 a_34
#        a_24 a_34 a_44  ]
# ordering the data like  (columns first)
#       [ *      *      a_13   a_24
#         *      a_12   a_23   a_34
#         a_11   a_22   a_33   a_44 ]
###


@inline inbands_getindex(A::Symmetric{<:Any, <:BandedMatrix}, k::Integer, j::Integer) =
    parent(A).data[A.k - abs(k-j) + 1, max(k,j)]



## eigvals routine

#
# function tridiagonalize!(A::SymBandedMatrix{T}) where {T}
#     n=size(A, 1)
#     d = Vector{T}(undef,n)
#     e = Vector{T}(undef,n-1)
#     q = Vector{T}(undef,0)
#     work = Vector{T}(undef,n)
#
#     sbtrd!('N','U',
#                 size(A,1),A.k,pointer(A),leadingdimension(A),
#                 pointer(d),pointer(e),pointer(q), n, pointer(work))
#
#     SymTridiagonal(d,e)
# end
#
#
# tridiagonalize(A::SymBandedMatrix) = tridiagonalize!(copy(A))
#
# eigvals!(A::SymBandedMatrix) = eigvals!(tridiagonalize!(A))
# eigvals(A::SymBandedMatrix) = eigvals!(copy(A))
#
# function eigvals!(A::SymBandedMatrix{T}, B::SymBandedMatrix{T}) where {T}
#     n = size(A, 1)
#     @assert n == size(B, 1)
#     # compute split-Cholesky factorization of B.
#     kb = bandwidth(B, 2)
#     ldb = leadingdimension(B)
#     pbstf!('U', n, kb, pointer(B), ldb)
#     # convert to a regular symmetric eigenvalue problem.
#     ka = bandwidth(A, 2)
#     lda = leadingdimension(A)
#     X = Array{T}(undef,0,0)
#     work = Vector{T}(undef,2n)
#     sbgst!('N', 'U', n, ka, kb, pointer(A), lda, pointer(B), ldb,
#            pointer(X), max(1, n), pointer(work))
#     # compute eigenvalues of symmetric eigenvalue problem.
#     eigvals!(A)
# end
#
# eigvals(A::SymBandedMatrix, B::SymBandedMatrix) = eigvals!(copy(A), copy(B))
#
#
# ## These routines give access to the necessary information to call BLAS
#
# @inline leadingdimension(B::SymBandedMatrix) = stride(B.data,2)
# @inline Base.pointer(B::SymBandedMatrix) = pointer(B.data)
