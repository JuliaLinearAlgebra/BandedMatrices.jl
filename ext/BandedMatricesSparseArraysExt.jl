module BandedMatricesSparseArraysExt

using BandedMatrices
using BandedMatrices: _banded_rowval, _banded_colval, _banded_nzval
using SparseArrays, FillArrays
import SparseArrays: sparse

function sparse(B::BandedMatrix)
	sparse(_banded_rowval(B), _banded_colval(B), _banded_nzval(B), size(B)...)
end

function BandedMatrices.bandwidths(A::SparseMatrixCSC)
    l = u = -max(size(A,1),size(A,2))
    n = size(A)[2]
    rows = rowvals(A)
    vals = nonzeros(A)

    if isempty(vals)
        return bandwidths(Zeros(1))
    end

    for j = 1:n
        for ind in nzrange(A, j)
            i = rows[ind]
            # We skip non-structural zeros when computing the
            # bandwidths.
            iszero(vals[ind]) && continue
            u = max(u, j-i)
            l = max(l, i-j)
        end
    end

    l,u
end

#Treat as n x 1 matrix
function BandedMatrices.bandwidths(A::SparseVector)
    l = u = -size(A,1)
    rows = rowvals(A)

    if isempty(rows)
        return bandwidths(Zeros(1))
    end

    for i in rows
        iszero(i) && continue
        u = max(u, 1-i)
        l = max(l, i-1)
    end
    
    l,u
end

end
