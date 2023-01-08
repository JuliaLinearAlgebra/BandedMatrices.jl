using Documenter, BandedMatrices

makedocs(;
    modules = [BandedMatrices],
    format = Documenter.HTML(
        canonical = "https://JuliaMatrices.github.io/BandedMatrices.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
    ],
    repo = "https://github.com/JuliaMatrices/BandedMatrices.jl/blob/{commit}{path}#L{line}",
    sitename = "BandedMatrices.jl",
    authors = "Sheehan Olver, Mikael Slevinsky, and contributors.",
)


deploydocs(;
    repo   = "github.com/JuliaMatrices/BandedMatrices.jl.git"
    )
