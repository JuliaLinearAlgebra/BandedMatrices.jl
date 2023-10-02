
using BandedMatrices
using LinearAlgebra

function diffs(a, b)
    n = length(a)
    d = Float64.(abs.(a - b))
    a1 = sum(d) / n
    r1 = sum(d ./ abs.(a)) / n
    a2 = sqrt(sum(d .* d) / n)
    r2 = sqrt(sum(d .* d ./ abs2.(a)) / n)
    a0 = maximum(d)
    r0 = maximum(d ./ abs.(a))
    a1, a2, a0, r1, r2, r0
end

function evaluate(n::Integer, T=Float64)

    M = BandedMatrix(0 => 6ones(n), 1 => -4ones(n-1), 2 => 1ones(n-2));
    MC = BandedMatrix(0 => 6ones(n), 1 => -4ones(n-1) .+ 0.1im, 2 => 1ones(n-2));
    M = T.(M)
    MC = Complex{T}.(MC)

    println("evaluation times with 5-banded test matrixes of dimension $n")
    println("LAPACK $T")
    Mb = @btime BandedMatrices.tridiagonalize!((Symmetric(copy($M))));

    println("generic $T")
    Mt = @btime tridiagonalize(Hermitian($M));

    println("generic with LinearAlgebra.givensAlgorithm $T")
    MtGA = @btime BandedMatrices.tridiagonalizeGA(Hermitian($M));

    println("generic Complex{$T}")
    Mtc = @btime tridiagonalize(Hermitian($MC));

    println("generic with LinearAlgebra.givensAlgorithm Complex{$T}")
    MtcGA = @btime BandedMatrices.tridiagonalizeGA(Hermitian($MC));


    println("generic BigFloat($(precision(BigFloat)))")
    Mbig = @time tridiagonalize(Hermitian(big.(M)));
    println("generic complex BigFloat($(precision(BigFloat)))")
    MCbig = @time tridiagonalize(Hermitian(big.(MC)));

    et = eigvals(Mt);
    etGA = eigvals(MtGA);
    etc = eigvals(Mtc);
    etcGA = eigvals(MtcGA);
    eb = eigvals(Mb);
    ebib = eigvals(Mbig);
    ebig = ebib;
    eCbig = eigvals(MCbig);

    println("absolute and relative errors")
    println("(mean / square-mean / max ) * (abs / rel)")

    println("LAPACK $T")
    println(diffs(eb, ebig))
    println("generic $T")
    println(diffs(et, ebig))
    println("generic with LinearAlgebra.givensAlgorithm $T")
    println(diffs(etGA, ebig))
    println("generic Complex{$T}")
    println(diffs(etc, eCbig))
    println("generic with LinearAlgebra.givensAlgorithm Complex{$T}")
    println(diffs(etcGA, eCbig))
end

evaluate( 1000);
