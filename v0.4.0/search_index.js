var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#BandedMatrices.jl-Documentation-1",
    "page": "Home",
    "title": "BandedMatrices.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#Creating-banded-matrices-1",
    "page": "Home",
    "title": "Creating banded matrices",
    "category": "section",
    "text": "BandedMatrixbonesbrandTo create a banded matrix of all zeros, identity matrix, or with a constant value use the following constructors:julia> BandedMatrix(Zeros(5,5), (1,2))\n5×5 BandedMatrices.BandedMatrix{Float64}:\n 0.0  0.0  0.0          \n 0.0  0.0  0.0  0.0     \n      0.0  0.0  0.0  0.0\n           0.0  0.0  0.0\n                0.0  0.0\n\njulia> BandedMatrix(Eye(5), (1,2))\n5×5 BandedMatrices.BandedMatrix{Float64}:\n 1.0  0.0  0.0          \n 0.0  1.0  0.0  0.0     \n      0.0  1.0  0.0  0.0\n           0.0  1.0  0.0\n                0.0  1.0\n\njulia> BandedMatrix(Ones(5,5), (1,2))\n5×5 BandedMatrices.BandedMatrix{Float64}:\n 1.0  1.0  1.0          \n 1.0  1.0  1.0  1.0     \n      1.0  1.0  1.0  1.0\n           1.0  1.0  1.0\n                1.0  1.0"
},

{
    "location": "index.html#Accessing-banded-matrices-1",
    "page": "Home",
    "title": "Accessing banded matrices",
    "category": "section",
    "text": "bandwidthsbandwidthbandrangebandBandRange"
},

{
    "location": "index.html#Creating-symmetric-banded-matrices-1",
    "page": "Home",
    "title": "Creating symmetric banded matrices",
    "category": "section",
    "text": "SymBandedMatrixsbonessbrand"
},

{
    "location": "index.html#Banded-matrix-interface-1",
    "page": "Home",
    "title": "Banded matrix interface",
    "category": "section",
    "text": "Banded matrices go beyond the type BandedMatrix: one can also create matrix types that conform to the _banded matrix interface_, in which case many of the utility functions in this package are available. The banded matrix interface consists of the following:Required methods Brief description\nbandwidth(A, k) Returns the sub-diagonal bandwidth (k==1) or the super-diagonal bandwidth (k==2)\nisbanded(A) Override to return true\ninbands_getindex(A, k, j) Unsafe: return A[k,j], without the need to check if we are inside the bands\ninbands_setindex!(A, v, k, j) Unsafe: set A[k,j] = v, without the need to check if we are inside the bandsNote that certain SubArrays of BandedMatrix are also banded matrices. The banded matrix interface is implemented for such SubArrays to take advantage of this."
},

{
    "location": "index.html#Implementation-1",
    "page": "Home",
    "title": "Implementation",
    "category": "section",
    "text": "Currently, only column-major ordering is supported: a banded matrix B[ a_11 a_12\n  a_21 a_22 a_23\n  a_31 a_32 a_33 a_34\n       a_42 a_43 a_44  ]is represented as a BandedMatrix with a field B.data representing the matrix as[ *     a_12   a_23    a_34\n a_11   a_22   a_33    a_44\n a_21   a_32   a_43    *\n a_31   a_42   *       *       ]B.l gives the number of subdiagonals (2) and B.u gives the number of super-diagonals (1).  Both B.l and B.u must be non-negative at the moment."
},

]}
