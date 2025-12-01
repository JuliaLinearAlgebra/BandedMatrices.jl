module DocstringTests

import BandedMatrices
using Documenter
using Test

if v"1.10" <= VERSION < v"1.11.0-"
    DocMeta.setdocmeta!(BandedMatrices, :DocTestSetup, :(using BandedMatrices))
    @testset "doctests" begin
        doctest(BandedMatrices)
    end
end

end
