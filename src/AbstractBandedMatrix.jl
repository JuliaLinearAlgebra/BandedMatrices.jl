# AbstractBandedMatrix must implement

abstract type AbstractBandedMatrix{T} <: AbstractSparseMatrix{T,Int} end

doc"""
    bandwidths(A)

Returns a tuple containing the upper and lower bandwidth of `A`.
"""
bandwidths(A::AbstractMatrix) = bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractMatrix) = -bandwidth(A,1),bandwidth(A,2)
bandinds(A::AbstractMatrix, k::Integer) = k==1 ? -bandwidth(A,1):bandwidth(A,2)


doc"""
    bandwidth(A,i)

Returns the lower bandwidth (`i==1`) or the upper bandwidth (`i==2`).
"""
bandwidth(A::AbstractMatrix, k::Integer) = k==1 ? size(A,1)-1 : size(A,2)-1

doc"""
    bandrange(A)

Returns the range `-bandwidth(A,1):bandwidth(A,2)`.
"""
bandrange(A::AbstractMatrix) = -bandwidth(A,1):bandwidth(A,2)



doc"""
    isbanded(A)

returns true if a matrix implements the banded interface.
"""
isbanded(::AbstractBandedMatrix) = true
isbanded(::) = false

# override bandwidth(A,k) for each AbstractBandedMatrix
# override inbands_getindex(A,k,j)


function Base.maximum(B::AbstractBandedMatrix)
    m=zero(eltype(B))
    for j = 1:size(B,2), k = colrange(B,j)
        m=max(B[k,j],m)
    end
    m
end

# ~ bound checking functions ~

checkbounds(A::AbstractBandedMatrix, k::Integer, j::Integer) =
    (0 < k ≤ size(A, 1) && 0 < j ≤ size(A, 2) || throw(BoundsError(A, (k,j))))

checkbounds(A::AbstractBandedMatrix, kr::Range, j::Integer) =
    (checkbounds(A, first(kr), j); checkbounds(A,  last(kr), j))

checkbounds(A::AbstractBandedMatrix, k::Integer, jr::Range) =
    (checkbounds(A, k, first(jr)); checkbounds(A, k,  last(jr)))

checkbounds(A::AbstractBandedMatrix, kr::Range, jr::Range) =
    (checkbounds(A, kr, first(jr)); checkbounds(A, kr,  last(jr)))

checkbounds(A::AbstractBandedMatrix, k::Colon, j::Integer) =
    (0 < j ≤ size(A, 2) || throw(BoundsError(A, (size(A,1),j))))

checkbounds(A::AbstractBandedMatrix, k::Integer, j::Colon) =
    (0 < k ≤ size(A, 1) || throw(BoundsError(A, (k,size(A,2)))))


# fallbacks for inbands_getindex and inbands_setindex!
@inline inbands_getindex(x::AbstractMatrix, i::Integer, j::Integer) = getindex(x, i, j)
@inline inbands_setindex!(x::AbstractMatrix, v, i::Integer, j::Integer) = setindex!(x, v, i, j)


## Show

type PrintShow
    str
end
Base.show(io::IO,N::PrintShow) = print(io,N.str)


showarray(io,M;opts...) = Base.showarray(io,M,false;opts...)
function Base.showarray(io::IO,B::AbstractBandedMatrix,repr::Bool = true; header = true)
    header && print(io,summary(B))

    if !isempty(B) && size(B,1) ≤ 1000 && size(B,2) ≤ 1000
        header && println(io,":")
        M=Array{Any}(size(B)...)
        fill!(M,PrintShow(""))
        for j = 1:size(B,2), k = colrange(B,j)
            M[k,j]=B[k,j]
        end

        showarray(io,M;header=false)
    end
end
