"""
    symrcm(matrix[, alg::Algorithm])

Use the Reverse Cuthill-Mckee algorithm to reduce the bandwith of a matrix.

```julia
julia> using BandedMatrices, CliqueTrees

julia> matrix = [
           0 1 1 0 0 0 0 0
           1 0 1 0 0 1 0 0
           1 1 0 1 1 0 0 0
           0 0 1 0 1 0 0 0
           0 0 1 1 0 0 1 1
           0 1 0 0 0 0 1 0
           0 0 0 0 1 1 0 1
           0 0 0 0 1 0 1 0
       ];

julia> bandedmatrix, perm, iperm = BandedMatrices.symrcm(matrix);

julia> bandedmatrix
8×8 LinearAlgebra.Symmetric{Int64, BandedMatrix{Int64, Matrix{Int64}, Base.OneTo{Int64}}}:
 0  1  1  0  ⋅  ⋅  ⋅  ⋅
 1  0  1  0  1  ⋅  ⋅  ⋅
 1  1  0  1  0  1  ⋅  ⋅
 0  0  1  0  0  1  0  ⋅
 ⋅  1  0  0  0  0  1  0
 ⋅  ⋅  1  1  0  0  1  1
 ⋅  ⋅  ⋅  0  1  1  0  1
 ⋅  ⋅  ⋅  ⋅  0  1  1  0

julia> bandedmatrix == matrix[perm, perm]
true
```
"""
function symrcm(matrix::AbstractMatrix)
    symrcm(matrix, Base.Sort.DEFAULT_UNSTABLE)
end
