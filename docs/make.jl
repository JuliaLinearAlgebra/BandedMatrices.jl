using Documenter, BandedMatrices

DocMeta.setdocmeta!(BandedMatrices, :DocTestSetup, :(using BandedMatrices); recursive=true)
makedocs(;
    modules = [BandedMatrices],
    pages = [
        "Home" => "index.md",
    ],
    sitename = "BandedMatrices.jl",
    authors = "Sheehan Olver, Mikael Slevinsky, and contributors.",
)


deploydocs(;
    repo   = "github.com/JuliaLinearAlgebra/BandedMatrices.jl.git"
    )
