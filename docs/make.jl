using Documenter, BandedMatrices

makedocs(modules=[BandedMatrices],
			doctest = true,
			clean = true,
			format = :html,
			sitename = "BandedMatrices.jl",
			authors = "Sheehan Olver",
			pages = Any[
					"Home" => "index.md"
					]
			)


deploydocs(
    repo   = "github.com/JuliaMatrices/BandedMatrices.jl.git",
    latest = "master",
    julia  = "1.0",
    osname = "linux",
    target = "build",
    deps   = nothing,
    make   = nothing
    )
