using BandedMatrices
using Test

import Aqua
@testset "Project quality" begin
    Aqua.test_all(BandedMatrices, ambiguities=false, piracies=false)
end

using Documenter
if v"1.10" <= VERSION < v"1.11.0-"
    DocMeta.setdocmeta!(BandedMatrices, :DocTestSetup, :(using BandedMatrices))
    @testset "doctests" begin
        doctest(BandedMatrices)
    end
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
