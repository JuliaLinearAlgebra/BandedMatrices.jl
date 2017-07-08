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


function Base.maximum(B::AbstractBandedMatrix)
    m=zero(eltype(B))
    for j = 1:size(B,2), k = colrange(B,j)
        m=max(B[k,j],m)
    end
    m
end


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
