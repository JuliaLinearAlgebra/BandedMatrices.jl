using BandedMatrices
using LinearAlgebra
using SparseArrays
using Test

using Aqua
@testset "Project quality" begin
    Aqua.test_all(BandedMatrices, ambiguities=false, piracy=false,
        # only test formatting on VERSION >= v1.7
        # https://github.com/JuliaTesting/Aqua.jl/issues/105#issuecomment-1551405866
        project_toml_formatting = VERSION >= v"1.7")
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
