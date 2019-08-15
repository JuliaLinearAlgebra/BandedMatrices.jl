using BandedMatrices, LinearAlgebra, LazyArrays, Test
    import LazyArrays: DenseColumnMajor
    import BandedMatrices: MemoryLayout, TriangularLayout, BandedColumns


@testset "Triangular" begin
    @testset "Upper" begin
        for (t, ud) in ((UpperTriangular, 'N'), (UnitUpperTriangular, 'U'))
            A = t(brand(10,10,1,2))
            @test isbanded(A)
            @test MemoryLayout(typeof(A)) == TriangularLayout{'U',ud,BandedColumns{DenseColumnMajor}}()
            @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (0,2)
            @test bandwidth(A,2) == 2
            @test BandedMatrix(A) == A
            @test A[2,1] == 0
            if( ud == 'U' )
                @test A[2,2] == 1
            end
            x=rand(10)
            @test A*x ≈ BandedMatrix(A)*x ≈ Matrix(A)*x
            @test transpose(A)*x ≈ transpose(BandedMatrix(A))*x ≈ transpose(Matrix(A))*x
            @test all(A*x .=== lmul!(A, copy(x)) .=== (similar(x) .= Mul(A,x)) .=== copyto!(similar(x), Mul(A,copy(x))) .===
                        BandedMatrices.tbmv!('U', 'N', ud, 10, 2, parent(A).data, copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbmv('U', 'N', ud, 10, 2, parent(A).data, rand(9))

            @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
            @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                        BandedMatrices.tbsv!('U', 'N', ud, 10, 2, parent(A).data, copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbsv('U', 'N', ud, 10, 2, parent(A).data, rand(9))
        end
    end
    @testset "Lower" begin
        for (t, ud) in ((LowerTriangular, 'N'), (UnitLowerTriangular, 'U'))
            A = t(brand(10,10,1,2))
            @test isbanded(A)
            @test MemoryLayout(typeof(A)) == TriangularLayout{'L', ud,BandedColumns{DenseColumnMajor}}()
            @test bandwidths(A) == bandwidths(BandedMatrix(A)) == (1,0)
            @test bandwidth(A,2) == 0
            @test BandedMatrix(A) == A
            @test A[1,2] == 0
            if( ud == 'U' )
                @test A[2,2] == 1
            end

            x=rand(10)
            @test A*x ≈ Matrix(A)*x
            @test transpose(A)*x ≈ transpose(BandedMatrix(A))*x ≈ transpose(Matrix(A))*x
            @test all(A*x .=== lmul!(A, copy(x)) .=== copyto!(similar(x), Mul(A,copy(x))) .=== (similar(x) .= Mul(A,x)) .===
                        BandedMatrices.tbmv!('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbmv('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), rand(9))

            @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
            @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                        BandedMatrices.tbsv!('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbsv('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), rand(9))
        end
    end
end
