using BandedMatrices
using ParallelTestRunner

# Start with autodiscovered tests
testsuite = find_tests(pwd())

if "--downstream_integration_test" in ARGS
    delete!(testsuite, "test_aqua")
end

filtered_args = filter(!=("--downstream_integration_test"), ARGS)
# Parse arguments
args = parse_args(filtered_args)

if filter_tests!(testsuite, args)
    delete!(testsuite, "benchmark")
    delete!(testsuite, "evaluations")
    delete!(testsuite, "mymatrix")
end

runtests(BandedMatrices, args; testsuite)
