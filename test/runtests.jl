using BandedMatrices
using LinearAlgebra
using SparseArrays
using Test

using Aqua
@testset "Project quality" begin
    Aqua.test_all(BandedMatrices, ambiguities=false, piracies=false)
end

using Documenter
DocMeta.setdocmeta!(BandedMatrices, :DocTestSetup, :(using BandedMatrices))
@testset "doctests" begin
    doctest(BandedMatrices)
end

include("test_banded.jl")
include("test_subarray.jl")
include("test_linalg.jl")
include("test_dot.jl")
include("test_broadcasting.jl")
include("test_indexing.jl")
include("test_bandedlu.jl")
include("test_bandedqr.jl")
include("test_symbanded.jl")
include("test_tribanded.jl")
include("test_interface.jl")
include("test_miscs.jl")
