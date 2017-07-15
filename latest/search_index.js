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
    "category": "Type",
    "text": "BandedMatrix(T, n, m, l, u)\n\nreturns an unitialized n×m banded matrix of type T with bandwidths (l,u).\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.beye",
    "page": "Home",
    "title": "BandedMatrices.beye",
    "category": "Function",
    "text": "beye(T,n,l,u)\n\nn×n banded identity matrix of type T with bandwidths (l,u)\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bones",
    "page": "Home",
    "title": "BandedMatrices.bones",
    "category": "Function",
    "text": "bones(T,n,m,l,u)\n\nCreates an n×m banded matrix  with ones in the bandwidth of type T with bandwidths (l,u)\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.brand",
    "page": "Home",
    "title": "BandedMatrices.brand",
    "category": "Function",
    "text": "brand(T,n,m,l,u)\n\nCreates an n×m banded matrix  with random numbers in the bandwidth of type T with bandwidths (l,u)\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bzeros",
    "page": "Home",
    "title": "BandedMatrices.bzeros",
    "category": "Function",
    "text": "bzeros(T,n,m,l,u)\n\nCreates an n×m banded matrix  of all zeros of type T with bandwidths (l,u)\n\n\n\n"
},

{
    "location": "index.html#Creating-banded-matrices-1",
    "page": "Home",
    "title": "Creating banded matrices",
    "category": "section",
    "text": "BandedMatrixbeyebonesbrandbzeros"
},

{
    "location": "index.html#BandedMatrices.bandwidths",
    "page": "Home",
    "title": "BandedMatrices.bandwidths",
    "category": "Function",
    "text": "bandwidths(A)\n\nReturns a tuple containing the upper and lower bandwidth of A.\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bandwidth",
    "page": "Home",
    "title": "BandedMatrices.bandwidth",
    "category": "Function",
    "text": "bandwidth(A,i)\n\nReturns the lower bandwidth (i==1) or the upper bandwidth (i==2).\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.bandrange",
    "page": "Home",
    "title": "BandedMatrices.bandrange",
    "category": "Function",
    "text": "bandrange(A)\n\nReturns the range -bandwidth(A,1):bandwidth(A,2).\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.band",
    "page": "Home",
    "title": "BandedMatrices.band",
    "category": "Function",
    "text": "band(i)\n\nRepresents the i-th band of a banded matrix.\n\njulia> using BandedMatrices\n\njulia> A = bones(5,5,1,1)\n5×5 BandedMatrices.BandedMatrix{Float64}:\n 1.0  1.0\n 1.0  1.0  1.0\n      1.0  1.0  1.0\n           1.0  1.0  1.0\n                1.0  1.0\n\njulia> A[band(1)]\n4-element Array{Float64,1}:\n 1.0\n 1.0\n 1.0\n 1.0\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.BandRange",
    "page": "Home",
    "title": "BandedMatrices.BandRange",
    "category": "Type",
    "text": "BandRange\n\nRepresents the entries in a row/column inside the bands.\n\njulia> using BandedMatrices\n\njulia> A = bones(5,5,1,1)\n5×5 BandedMatrices.BandedMatrix{Float64}:\n 1.0  1.0\n 1.0  1.0  1.0\n      1.0  1.0  1.0\n           1.0  1.0  1.0\n                1.0  1.0\n\njulia> A[2,BandRange]\n3-element Array{Float64,1}:\n 1.0\n 1.0\n 1.0\n\n\n\n"
},

{
    "location": "index.html#Accessing-banded-matrices-1",
    "page": "Home",
    "title": "Accessing banded matrices",
    "category": "section",
    "text": "bandwidthsbandwidthbandrangebandBandRange"
},

{
    "location": "index.html#BandedMatrices.SymBandedMatrix",
    "page": "Home",
    "title": "BandedMatrices.SymBandedMatrix",
    "category": "Type",
    "text": "SymBandedMatrix(T, n, k)\n\nreturns an unitialized n×n symmetric banded matrix of type T with bandwidths (-k,k).\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.sbeye",
    "page": "Home",
    "title": "BandedMatrices.sbeye",
    "category": "Function",
    "text": "sbeye(T,n,l,u)\n\nn×n banded identity matrix of type T with bandwidths (l,u)\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.sbrand",
    "page": "Home",
    "title": "BandedMatrices.sbrand",
    "category": "Function",
    "text": "sbrand(T,n,k)\n\nCreates an n×n symmetric banded matrix  with random numbers in the bandwidth of type T with bandwidths (k,k)\n\n\n\n"
},

{
    "location": "index.html#BandedMatrices.sbzeros",
    "page": "Home",
    "title": "BandedMatrices.sbzeros",
    "category": "Function",
    "text": "sbzeros(T,n,k)\n\nCreates an n×n symmetric banded matrix  of all zeros of type T with bandwidths (k,k)\n\n\n\n"
},

{
    "location": "index.html#Creating-symmetric-banded-matrices-1",
    "page": "Home",
    "title": "Creating symmetric banded matrices",
    "category": "section",
    "text": "SymBandedMatrixsbeyesbonessbrandsbzeros"
},

{
    "location": "index.html#Banded-matrix-interface-1",
    "page": "Home",
    "title": "Banded matrix interface",
    "category": "section",
    "text": "Banded matrices go beyond the type BandedMatrix: one can also create matrix types that conform to the _banded matrix interface_, in which case many of the utility functions in this package are available. The banded matrix interface consists of the following: | Required methods | Brief description | | –––––––- | –––––––- | | bandwidth(A, k) | Returns the sub-diagonal bandwidth (k==1) or the super-diagonal bandwidth (k==2) | | isbanded(A)    | Override to return true | | inbands_getindex(A, k, j) | Unsafe: return A[k,j], without the need to check if we are inside the bands | | inbands_setindex!(A, v, k, j) | Unsafe: set A[k,j] = v, without the need to check if we are inside the bands | | ––––––– | ––––––- |"
},

]}
