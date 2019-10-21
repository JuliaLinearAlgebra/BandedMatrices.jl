using BandedMatrices, LazyArrays, LinearAlgebra, FillArrays, Test
import Base.Broadcast: materialize, broadcasted
import BandedMatrices: BandedColumns
import LazyArrays: SymmetricLayout, MemoryLayout, Applied, MulAddStyle, DenseColumnMajor

@testset "Linear Algebra" begin
    @testset "Matrix types" begin
        A = brand(5,5,1,2)
        x = randn(5)

        @test A*x ≈ Matrix(A)*x

        @test A*A == A^2 ≈ Matrix(A)^2
        @test A*A isa BandedMatrix
        @test A^2 isa BandedMatrix
        @test bandwidths(A^2) == (2,4)

        @test A*A' isa BandedMatrix
        @test A*A' ≈ Matrix(A)*Matrix(A)'
        @test bandwidths(A*A') == (3,3)

        @test A'*A isa BandedMatrix
        @test A'*A ≈ Matrix(A)'*Matrix(A)
        @test bandwidths(A'*A) == (3,3)

        @test applied(*,Symmetric(A),A) isa Applied{MulAddStyle}
        @test MemoryLayout(typeof(Symmetric(A))) == SymmetricLayout{BandedColumns{DenseColumnMajor}}()
        @test Symmetric(A)*A isa BandedMatrix
        @test Symmetric(A)*A ≈ Symmetric(Matrix(A))*Matrix(A)
        @test bandwidths(Symmetric(A)*A) == (3,4)

        @test Hermitian(A)*A isa BandedMatrix
        @test Hermitian(A)*A ≈ Hermitian(Matrix(A))*Matrix(A)
        @test bandwidths(Hermitian(A)*A) == (3,4)

        B = A+im*A
        @test Hermitian(B)*A isa BandedMatrix
        @test Hermitian(B)*A ≈ Hermitian(Matrix(B))*Matrix(A)
        @test bandwidths(Hermitian(B)*A) == (3,4)

        @test UpperTriangular(A)*A isa BandedMatrix
        @test UpperTriangular(A)*A ≈ UpperTriangular(Matrix(A))*Matrix(A)
        @test bandwidths(UpperTriangular(A)*A) == (1,4)

        B = apply(*,A,A,A)
        @test B isa BandedMatrix
        @test all(B .=== (A*A)*A)
        @test bandwidths(B) == (3,4)
    end

    @testset "gbmm!" begin
        @testset "gbmm! subpieces step by step and column by column" begin 
            for n in (1,5,50), ν in (1,5,50), m in (1,5,50),
                            Al in (0,1,2,30), Au in (0,1,2,30),
                            Bl in (0,1,2,30), Bu in (0,1,2,30)
                A=brand(n,ν,Al,Au)
                B=brand(ν,m,Bl,Bu)
                α,β,T=0.123,0.456,Float64
                C=brand(Float64,n,m,A.l+B.l,A.u+B.u)
                a=pointer(A.data)
                b=pointer(B.data)
                c=pointer(C.data)
                sta=max(1,stride(A.data,2))
                stb=max(1,stride(B.data,2))
                stc=max(1,stride(C.data,2))

                sz=sizeof(T)

                mr=1:min(m,1+B.u)
                exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
                for j=mr
                    BandedMatrices.A11_Btop_Ctop_gbmv!(α,β,
                                                    n,ν,m,j,
                                                    sz,
                                                    a,A.l,A.u,sta,
                                                    b,B.l,B.u,stb,
                                                    c,C.l,C.u,stc)
                end
                @test C[:,mr] ≈ exC[:,mr]

                mr=1+B.u:min(1+C.u,ν+B.u,m)
                exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
                for j=mr
                    BandedMatrices.Atop_Bmid_Ctop_gbmv!(α,β,
                                                    n,ν,m,j,
                                                    sz,
                                                    a,A.l,A.u,sta,
                                                    b,B.l,B.u,stb,
                                                    c,C.l,C.u,stc)
                end
                if !isempty(mr)
                    @test C[:,mr] ≈ exC[:,mr]
                end

                mr=1+C.u:min(m,ν+B.u,n+C.u)
                exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
                for j=mr
                    BandedMatrices.Amid_Bmid_Cmid_gbmv!(α,β,
                                                    n,ν,m,j,
                                                    sz,
                                                    a,A.l,A.u,sta,
                                                    b,B.l,B.u,stb,
                                                    c,C.l,C.u,stc)
                end
                if !isempty(mr)
                    @test C[:,mr] ≈ exC[:,mr]
                end

                mr=ν+B.u+1:min(m,n+C.u)
                exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
                for j=mr
                    BandedMatrices.Anon_Bnon_C_gbmv!(α,β,
                                                    n,ν,m,j,
                                                    sz,
                                                    a,A.l,A.u,sta,
                                                    b,B.l,B.u,stb,
                                                    c,C.l,C.u,stc)
                end
                if !isempty(mr)
                    @test C[:,mr] ≈ exC[:,mr]
                end
            end
        end

        for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
            A=brand(n,ν,Al,Au)
            B=brand(ν,m,Bl,Bu)
            α,β,T=0.123,0.456,Float64
            C=brand(Float64,n,m,A.l+B.l,A.u+B.u)
            Cold = deepcopy(C)
            exC=α*Matrix(A)*Matrix(B)+β*Matrix(C)
            BandedMatrices.gbmm!('N','N', α,A,B,β,C)

            @test Matrix(exC) ≈ Matrix(C)
        end
    end

    @testset "Negative bands fills with zero" begin
        A = brand(10,10,2,2)
        B = brand(10,10,-2,2)
        C = BandedMatrix(Fill(NaN,10,10),(0,4))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(10,10,-2,2)
        B = brand(10,10,-2,2)
        C = BandedMatrix(Fill(NaN,10,10),(-4,4))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(10,10,-2,2)
        B = brand(10,10,2,2)
        C = BandedMatrix(Fill(NaN,10,10),(0,4))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(10,10,2,2)
        B = brand(10,10,2,-2)
        C = BandedMatrix(Fill(NaN,10,10),(4,0))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(10,10,2,-2)
        B = brand(10,10,2,-2)
        C = BandedMatrix(Fill(NaN,10,10),(4,-4))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(10,10,2,-2)
        B = brand(10,10,2,2)
        C = BandedMatrix(Fill(NaN,10,10),(4,0))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)

        A = brand(30,1,0,0)
        B = brand(1,30,17,17)
        C = BandedMatrix(Fill(NaN, 30,30), (17,17))
        C .= Mul(A,B)
        @test C ≈ Matrix(A)*Matrix(B)
    end

    @testset "Not enough bands" begin
        A = BandedMatrix(Zeros(10,10), (1,1))
        A[band(0)] .= randn(10)
        B = BandedMatrix(randn(10,10), (1,1))
        C = BandedMatrix(Zeros(10,10), (1,1))

        C .= Mul(A,B)

        @test all(C .=== A*B)

        A[band(1)] .= randn(9)
        @test_throws BandError C .= Mul(A,B)

        A = BandedMatrix(randn(2,1), (2,0))
        A[1,1] = 0
        B = BandedMatrix(randn(1,1), (1,1))
        C = BandedMatrix(randn(2,1), (2,0))
        D = A * B + C
        @test BandedMatrices.gbmm!('N','N', 1.0 , A, B, 1.0, C) ≈ D
    end

    @testset "BandedMatrix{Int} * Vector{Vector{Int}}" begin
        A, x =  [1 2; 3 4] , [[1,2],[3,4]]
        @test BandedMatrix(A)*x == A*x
    end

    @testset "Sym * Banded bug" begin
        A = SymTridiagonal(randn(10),randn(9))
        B = BandedMatrix((-1 => Ones{Int}(8),), (10,8))

        M = Mul(A,B)
        C = Matrix{Float64}(undef,10,8)

        C .= M
        @test C == A*B == Matrix(A)*Matrix(B)
        @test A*B isa BandedMatrix
    end

    @testset "Overwrite NaN" begin
        B = BandedMatrix{Float64}(undef,(2,2),(-1,-1))
        @test copyto!(fill(NaN,2), Mul(B,ones(2))) == [0.0,0.0]
    end

    @testset "NaN Bug" begin
        C = BandedMatrix{Float64}(undef, (1,2), (0,2)); C.data .= NaN;
        A = brand(1,1,0,1)
        B = brand(1,2,0,2)
        C .= Mul(A,B)
        @test C == A*B

        C.data .= NaN
        C .= @~ 1.0 * A*B + 0.0 * C
        @test C == A*B
    end

    @testset "x' ambiguity (#102)" begin
        x = randn(10)
        A = brand(10,10,1,1)
        @test x'A ≈ x'Matrix(A) ≈ transpose(x)A
    end

    @testset "mismatched dimensions (#118)" begin
        m = BandedMatrix(Eye(3), (0,0))
        @test_throws DimensionMismatch m * [1,2]
        @test_throws DimensionMismatch m * [1, 2, 3, 4]
        @test_throws DimensionMismatch m \ [1, 2]
    end

    @testset "Banded*Diagonal*Banded" begin
        A = brand(4,4,1,1)
        D= Diagonal(randn(4))
        @test A*D*A isa BandedMatrix
        @test Matrix(A)*D*Matrix(A) ≈ A*D*A
        @test D*A' isa BandedMatrix
        @test A'D isa BandedMatrix
        @test A'*D*A isa BandedMatrix
        @test  A'*D*A ≈ Matrix(A)'*D*Matrix(A)
    end
end

