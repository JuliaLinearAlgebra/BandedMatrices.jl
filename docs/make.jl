using Documenter, BandedMatrices

makedocs()

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/JuliaMatrices/BandedMatrices.jl.git",
    julia  = "0.5",
    osname = "osx"
)
