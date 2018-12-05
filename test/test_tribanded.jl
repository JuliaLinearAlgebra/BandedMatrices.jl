using BandedMatrices, LinearAlgebra, LazyArrays, Test
    import LazyArrays: DenseColumnMajor
    import BandedMatrices: MemoryLayout, TriangularLayout, BandedColumns


@testset "Triangular" begin
    @testset "Upper" begin
        A = UpperTriangular(brand(10,10,1,2))
        @test isbanded(A)
        @test MemoryLayout(A) == TriangularLayout{'U','N'}(BandedColumns(DenseColumnMajor()))
        @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (0,2)
        @test bandwidth(A,2) == 2
        @test BandedMatrix(A) == A
        @test A[2,1] == 0

        x=rand(10)
        @test A*x ≈ BandedMatrix(A)*x ≈ Matrix(A)*x
        @test all(A*x .=== lmul!(A, copy(x)) .=== (similar(x) .= Mul(A,x)) .=== copyto!(similar(x), Mul(A,copy(x))) .===
                    BandedMatrices.tbmv!('U', 'N', 'N', 10, 2, parent(A).data, copy(x)))

        @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
        @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                    BandedMatrices.tbsv!('U', 'N', 'N', 10, 2, parent(A).data, copy(x)))

        A = UnitUpperTriangular(brand(10,10,1,2))
        @test isbanded(A)
        @test MemoryLayout(A) == TriangularLayout{'U','U'}(BandedColumns(DenseColumnMajor()))
        @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (0,2)
        @test bandwidth(A,2) == 2
        @test BandedMatrix(A) == A
        @test A[2,1] == 0
        @test A[2,2] == 1

        x=rand(10)
        @test A*x ≈ Matrix(A)*x
        @test all(A*x .=== lmul!(A, copy(x)) .=== (similar(x) .= Mul(A,x)) .=== copyto!(similar(x), Mul(A,copy(x))) .===
                    BandedMatrices.tbmv!('U', 'N', 'U', 10, 2, parent(A).data, copy(x)))

        @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
        @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                    BandedMatrices.tbsv!('U', 'N', 'U', 10, 2, parent(A).data, copy(x)))
    end
    @testset "Lower" begin
        A = LowerTriangular(brand(10,10,1,2))
        @test isbanded(A)
        @test MemoryLayout(A) == TriangularLayout{'L','N'}(BandedColumns(DenseColumnMajor()))
        @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,0)
        @test bandwidth(A,2) == 0
        @test BandedMatrix(A) == A
        @test A[1,2] == 0

        x=rand(10)
        @test A*x ≈ Matrix(A)*x
        @test all(A*x .=== lmul!(A, copy(x)) .=== copyto!(similar(x), Mul(A,copy(x))) .=== (similar(x) .= Mul(A,x)) .===
                    BandedMatrices.tbmv!('L', 'N', 'N', 10, 1, view(parent(A).data, 3:4,:), copy(x)))

        @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
        @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                    BandedMatrices.tbsv!('L', 'N', 'N', 10, 1, view(parent(A).data, 3:4,:), copy(x)))

        A = UnitLowerTriangular(brand(10,10,1,2))
        @test isbanded(A)
        @test MemoryLayout(A) == TriangularLayout{'L','U'}(BandedColumns(DenseColumnMajor()))
        @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,0)
        @test bandwidth(A,2) == 0
        @test BandedMatrix(A) == A
        @test A[1,2] == 0
        @test A[2,2] == 1

        x=rand(10)
        @test A*x ≈ Matrix(A)*x
        @test all(A*x .=== lmul!(A, copy(x)) .=== copyto!(similar(x), Mul(A,copy(x))) .=== (similar(x) .= Mul(A,x)) .===
                    BandedMatrices.tbmv!('L', 'N', 'U', 10, 1, view(parent(A).data, 3:4,:), copy(x)))

        @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
        @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                    BandedMatrices.tbsv!('L', 'N', 'U', 10, 1, view(parent(A).data, 3:4,:), copy(x)))
    end
end
