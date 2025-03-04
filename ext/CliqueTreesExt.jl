module CliqueTreesExt

using BandedMatrices
using Base.Sort: Algorithm
using CliqueTrees: CliqueTrees, permutation, RCMMD
using CliqueTrees.SparseArrays
using LinearAlgebra

function BandedMatrices.symrcm(matrix::AbstractMatrix, alg::Algorithm)
    BandedMatrices.symrcm(sparse(matrix), alg)
end

function BandedMatrices.symrcm(matrix::SparseMatrixCSC{T}, alg::Algorithm) where T
    order, index = permutation(matrix; alg=RCMMD(alg))
    lower = tril!(permute(matrix, order, order))
    
    bandwidth = maximum(axes(lower, 2)) do j
        p = last(nzrange(lower, j))
        return rowvals(lower)[p] - j
    end
    
    banded = BandedMatrix{T}(undef, size(lower), (bandwidth, 0))
    fill!(banded.data, zero(T))
    
    @inbounds for j in axes(lower, 2)
        for p in nzrange(lower, j)
            i = rowvals(lower)[p]
            banded.data[i - j + 1, j] = nonzeros(lower)[p]
        end
    end
    
    return Symmetric(banded, :L), order, index
end

end
