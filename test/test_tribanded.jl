module TestTriBanded

using BandedMatrices, LinearAlgebra, ArrayLayouts, Test
import BandedMatrices: BandedColumns, BandedRows, isbanded


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
            @test all(A*x .≈ lmul!(A, copy(x)) .=== lmul(A,x) .===
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
            @test all(A*x .≈ lmul!(A, copy(x)) .=== lmul(A,x) .===
                        BandedMatrices.tbmv!('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbmv('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), rand(9))

            @test A\x ≈ Matrix(A) \ x ≈ BandedMatrix(A) \ x
            @test all((A \ x) .=== ldiv!(A, copy(x)) .=== copyto!(similar(x), Ldiv(A, x)) .=== (similar(x) .= Ldiv(A, x)) .===
                        BandedMatrices.tbsv!('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), copy(x)))
            @test_throws DimensionMismatch BandedMatrices.tbsv('L', 'N', ud, 10, 1, view(parent(A).data, 3:4,:), rand(9))
        end
    end

    @testset "subarray" begin
        A = brand(6,6,2,1)
        L = LowerTriangular(A)
        U = UpperTriangular(A)
        V = view(L,2:4,1:3)
        @test MemoryLayout(V) isa BandedColumns{DenseColumnMajor}
        @test L[2:4,1:3] == BandedMatrix(V) == LowerTriangular(Matrix(A))[2:4,1:3]
        @test L[2:4,1:3] isa BandedMatrix
        @test bandwidths(L[2:4,1:3]) == bandwidths(V) == (1,1)
        @test U[2:4,1:3] == UpperTriangular(Matrix(A))[2:4,1:3]
        @test U[2:4,1:3] isa BandedMatrix
        @test bandwidths(U[2:4,1:3]) == (-1,2)
    end

    @testset "row major" begin
        A = brand(5,5,2,1)
        b = randn(5)
        U = UpperTriangular(A')
        L = LowerTriangular(A')
        Ũ = LowerTriangular(A)'
        L̃ = UpperTriangular(A)'
        @test MemoryLayout(U) isa TriangularLayout{'U','N',BandedRows{DenseColumnMajor}}
        @test U*b ≈ ArrayLayouts.lmul!(U,copy(b)) ≈ Matrix(U)*b
        @test L*b ≈ ArrayLayouts.lmul!(L,copy(b)) ≈ Matrix(L)*b
        @test ldiv!(U,copy(b)) ≈ ArrayLayouts.ldiv!(U,copy(b)) ≈ U\b ≈ Matrix(U)\b
        @test ldiv!(L,copy(b)) ≈ ArrayLayouts.ldiv!(L,copy(b)) ≈ L\b ≈ Matrix(L)\b
        @test ldiv!(Ũ,copy(b)) ≈ ArrayLayouts.ldiv!(Ũ,copy(b)) ≈ Ũ\b ≈ Matrix(Ũ)\b
        @test ldiv!(L̃,copy(b)) ≈ ArrayLayouts.ldiv!(L̃,copy(b)) ≈ L̃\b ≈ Matrix(L̃)\b

        B = randn(5,5)

        @test U*B ≈ ArrayLayouts.lmul!(U,copy(B)) ≈ Matrix(U)*B
        @test L*B ≈ ArrayLayouts.lmul!(L,copy(B)) ≈ Matrix(L)*B
        @test B*U ≈ ArrayLayouts.rmul!(copy(B),U) ≈ B*Matrix(U)
        @test B*L ≈ ArrayLayouts.rmul!(copy(B),L) ≈ B*Matrix(L)

        @test U \ B ≈ Matrix(U) \ B
        @test B / U ≈ B / Matrix(U)
        @test L \ B ≈ Matrix(L) \ B
        @test B / L ≈ B / Matrix(L)
        @test Ũ \ B ≈ Matrix(Ũ) \ B
        @test B / Ũ ≈ B / Matrix(Ũ)
        @test L̃ \ B ≈ Matrix(L̃) \ B
        @test B / L̃ ≈ B / Matrix(L̃)
    end

    @testset "rot180" begin
        B = brand(5,5,2,1)
        @test rot180(UpperTriangular(B)) isa LowerTriangular{<:Any,<:BandedMatrix}
        @test rot180(UnitUpperTriangular(B)) isa UnitLowerTriangular{<:Any,<:BandedMatrix}
        @test rot180(LowerTriangular(B)) isa UpperTriangular{<:Any,<:BandedMatrix}
        @test rot180(UnitLowerTriangular(B)) isa UnitUpperTriangular{<:Any,<:BandedMatrix}
        @test rot180(UpperTriangular(B)) == rot180(Matrix(UpperTriangular(B)))
        @test rot180(UnitUpperTriangular(B)) == rot180(Matrix(UnitUpperTriangular(B)))
        @test rot180(LowerTriangular(B)) == rot180(Matrix(LowerTriangular(B)))
        @test rot180(UnitLowerTriangular(B)) == rot180(Matrix(UnitLowerTriangular(B)))
    end

    @testset "empty uppertriangular" begin
        B = brand(0,0,2,1)
        @test UpperTriangular(B) \ Float64[] == Float64[]
    end
end

end # module
