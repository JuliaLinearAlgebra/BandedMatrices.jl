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
    "location": "index.html#BandedMatrices.BandedMatrix",
    "page": "Home",
    "title": "BandedMatrices.BandedMatrix",
    "category": "type",
    "text": "BandedMatrix{T}(undef, (n, m), (l, u))\n\nreturns an uninitialized n×m banded matrix of type T with bandwidths (l,u).\n\n\n\n\n\nBandedMatrix{T}(kv::Pair, (m,n), (l,u))\n\nConstruct a m × n BandedMatrix with bandwidths (l,u) from Pairs of diagonals and vectors. Vector kv.second will be placed on the kv.first diagonal.\n\n\n\n\n\nBandedMatrix(kv::Pair{<:Integer,<:AbstractVector}...)\n\nConstruct a square matrix from Pairs of diagonals and vectors. Vector kv.second will be placed on the kv.first diagonal.\n\n\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.brand",
    "page": "Home",
    "title": "BandedMatrices.brand",
    "category": "function",
    "text": "brand(T,n,m,l,u)\n\nCreates an n×m banded matrix  with random numbers in the bandwidth of type T with bandwidths (l,u)\n\n\n\n\n\n"
},

{
    "location": "index.html#Creating-banded-matrices-1",
    "page": "Home",
    "title": "Creating banded matrices",
    "category": "section",
    "text": "BandedMatrixbonesbrandDocTestSetup = quote\n    using BandedMatrices\nendTo create a banded matrix of all zeros, identity matrix, or with a constant value use the following constructors:julia> BandedMatrix(Zeros(5,5), (1,2))\n5×5 BandedMatrix{Float64,Array{Float64,2}}:\n 0.0  0.0  0.0   ⋅    ⋅\n 0.0  0.0  0.0  0.0   ⋅\n  ⋅   0.0  0.0  0.0  0.0\n  ⋅    ⋅   0.0  0.0  0.0\n  ⋅    ⋅    ⋅   0.0  0.0\n\njulia> BandedMatrix(Eye(5), (1,2))\n5×5 BandedMatrix{Float64,Array{Float64,2}}:\n 1.0  0.0  0.0   ⋅    ⋅\n 0.0  1.0  0.0  0.0   ⋅\n  ⋅   0.0  1.0  0.0  0.0\n  ⋅    ⋅   0.0  1.0  0.0\n  ⋅    ⋅    ⋅   0.0  1.0\n\njulia> BandedMatrix(Ones(5,5), (1,2))\n5×5 BandedMatrix{Float64,Array{Float64,2}}:\n 1.0  1.0  1.0   ⋅    ⋅\n 1.0  1.0  1.0  1.0   ⋅\n  ⋅   1.0  1.0  1.0  1.0\n  ⋅    ⋅   1.0  1.0  1.0\n  ⋅    ⋅    ⋅   1.0  1.0To create a banded matrix of a given size with constant bands (such as the classical finite difference approximation of the one-dimensional Laplacian on the unit interval [0,1]), you can use the following:n = 128\nh = 1/n\nA = BandedMatrix{Float64}(undef, (n,n), (1,1))\nA[band(0)] .= -2/h^2\nA[band(1)] .= A[band(-1)] .= 1/h^2"
},

{
    "location": "index.html#BandedMatrices.bandwidths",
    "page": "Home",
    "title": "BandedMatrices.bandwidths",
    "category": "function",
    "text": "bandwidths(A)\n\nReturns a tuple containing the upper and lower bandwidth of A.\n\n\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bandwidth",
    "page": "Home",
    "title": "BandedMatrices.bandwidth",
    "category": "function",
    "text": "bandwidth(A,i)\n\nReturns the lower bandwidth (i==1) or the upper bandwidth (i==2).\n\n\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bandrange",
    "page": "Home",
    "title": "BandedMatrices.bandrange",
    "category": "function",
    "text": "bandrange(A)\n\nReturns the range -bandwidth(A,1):bandwidth(A,2).\n\n\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.band",
    "page": "Home",
    "title": "BandedMatrices.band",
    "category": "function",
    "text": "band(i)\n\nRepresents the i-th band of a banded matrix.\n\njulia> using BandedMatrices\n\njulia> A = BandedMatrix(Ones(5,5),(1,1))\n5×5 BandedMatrix{Float64,Array{Float64,2}}:\n 1.0  1.0   ⋅    ⋅    ⋅\n 1.0  1.0  1.0   ⋅    ⋅\n  ⋅   1.0  1.0  1.0   ⋅\n  ⋅    ⋅   1.0  1.0  1.0\n  ⋅    ⋅    ⋅   1.0  1.0\n\njulia> A[band(1)]\n4-element Array{Float64,1}:\n 1.0\n 1.0\n 1.0\n 1.0\n\n\n\n\n\n"
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
    "text": "Use Symmetric(::BandedMatrix) to work with symmetric banded matrices."
},

{
    "location": "index.html#Banded-matrix-interface-1",
    "page": "Home",
    "title": "Banded matrix interface",
    "category": "section",
    "text": "Banded matrices go beyond the type BandedMatrix: one can also create matrix types that conform to the banded matrix interface, in which case many of the utility functions in this package are available. The banded matrix interface consists of the following:Required methods Brief description\nbandwidths(A) Returns a tuple containing the sub-diagonal and super-diagonal bandwidth\nisbanded(A) Override to return true\n:–––––––- :–––––––-\nOptional methods Brief description\n:–––––––- :–––––––-\ninbands_getindex(A, k, j) Unsafe: return A[k,j], without the need to check if we are inside the bands\ninbands_setindex!(A, v, k, j) Unsafe: set A[k,j] = v, without the need to check if we are inside the bands\nBandedMatrices.MemoryLayout(A) Override to get banded lazy linear algebra, e.g. y .= Mul(A,x)\nBandedMatrices.bandeddata(A) Override to return a matrix of the entries in BLAS format. Required if MemoryLayout(A) returns BandedColumnMajorNote that certain SubArrays of BandedMatrix are also banded matrices. The banded matrix interface is implemented for such SubArrays to take advantage of this."
},

{
    "location": "index.html#Implementation-1",
    "page": "Home",
    "title": "Implementation",
    "category": "section",
    "text": "Currently, only column-major ordering is supported: a banded matrix B[ a_11 a_12  ⋅    ⋅\n  a_21 a_22 a_23  ⋅\n  a_31 a_32 a_33 a_34\n   ⋅   a_42 a_43 a_44  ]is represented as a BandedMatrix with a field B.data representing the matrix as[ ⋅     a_12   a_23    a_34\n a_11   a_22   a_33    a_44\n a_21   a_32   a_43    ⋅\n a_31   a_42   ⋅       ⋅       ]B.l gives the number of subdiagonals (2) and B.u gives the number of super-diagonals (1).  Both B.l and B.u must be non-negative at the moment."
},

]}
