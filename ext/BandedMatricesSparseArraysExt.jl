module BandedMatricesSparseArraysExt

using BandedMatrices
using BandedMatrices: _banded_rowval, _banded_colval, _banded_nzval
using SparseArrays
import SparseArrays: sparse

function sparse(B::BandedMatrix)
	sparse(_banded_rowval(B), _banded_colval(B), _banded_nzval(B), size(B)...)
end

function BandedMatrices.bandwidths(A::SparseMatrixCSC)
    l = u = -max(size(A,1),size(A,2))

    n = size(A)[2]
    rows = rowvals(A)
    vals = nonzeros(A)
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

function BandedMatrices.bandwidths(A::SparseVector)
    l = u = -size(A,1)

    rows = rowvals(A)
    for i in rows
        # We skip non-structural zeros when computing the
        # bandwidths.
        iszero(i) && continue
        u = max(u, 1-i)
        l = max(l, i-1)
    end
    l,u
end

end
