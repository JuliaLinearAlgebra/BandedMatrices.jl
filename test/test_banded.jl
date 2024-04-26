module TestBanded

using ArrayLayouts
using BandedMatrices
import BandedMatrices: _BandedMatrix
using FillArrays
using LinearAlgebra
using SparseArrays
using Test

include("mymatrix.jl")

@testset "BandedMatrix" begin
    @testset "Undef BandedMatrix" begin
        @test typeof(BandedMatrix{Float64}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Float64,Matrix{Float64}}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Float64,Matrix{Float64},Base.OneTo{Int}}(undef,(10,10),(1,1)))
        @test typeof(BandedMatrix{Int}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Int,Matrix{Int}}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Int,Matrix{Int},Base.OneTo{Int}}(undef,(10,10),(1,1)))
        @test typeof(BandedMatrix{Vector{Int}}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Vector{Int},Matrix{Vector{Int}}}(undef,(10,10),(1,1))) ==
            typeof(BandedMatrix{Vector{Int},Matrix{Vector{Int}},Base.OneTo{Int}}(undef,(10,10),(1,1)))
    end

    @testset "Creating BandedMatrix" begin
        @test BandedMatrix(Zeros(5,5), (1,1)) == _BandedMatrix(zeros(3,5), 5, 1, 1)
        @test BandedMatrix(Zeros{Int32}(5,5), (1,1)) isa BandedMatrix{Int32, Matrix{Int32}}
        @test BandedMatrix(Zeros{Int}(5,5), (1,1)) == _BandedMatrix(zeros(Int,3,5), 5, 1, 1)
        @test BandedMatrix{Int}(Zeros(5,5), (1,1)) == _BandedMatrix(zeros(Int,3,5), 5, 1, 1)
        @test Matrix(BandedMatrix(Ones(5,5), (1,1))) == Matrix(BandedMatrix(Fill(1.0,(5,5)), (1,1))) ==
                                                        Matrix(SymTridiagonal(ones(5), ones(4)))

        @test all(BandedMatrix(0 => 1:5, 2=> 2:3, -3=> 1:7) .=== diagm(0 => 1:5, 2=> 2:3, -3=> 1:7))
        @test all(BandedMatrix{Float64}(0 => 1:5, 2=> 2:3, -3=> 1:7) .=== Matrix{Float64}(diagm(0 => 1:5, 2=> 2:3, -3=> 1:7)))
        @test all(BandedMatrix((0 => 1:5, 2=> 2:3, -3=> 1:7),(10,10)) .=== Matrix{Int}(diagm(0 => 1:5, 2=> 2:3, -3=> 1:7)))
        @test all(BandedMatrix{Float64}((0 => 1:5, 2=> 2:3, -3=> 1:7),(10,10)) .=== Matrix{Float64}(diagm(0 => 1:5, 2=> 2:3, -3=> 1:7)))
        @test all(BandedMatrix((0 => 1:5, 2=> 2:3, -3=> 1:7),(10,10),(4,3)) .=== Matrix{Int}(diagm(0 => 1:5, 2=> 2:3, -3=> 1:7)))
        @test all(BandedMatrix{Float64}((0 => 1:5, 2=> 2:3, -3=> 1:7),(10,10),(4,3)) .=== Matrix{Float64}(diagm(0 => 1:5, 2=> 2:3, -3=> 1:7)))

        matrix = MyMatrix(ones(Int64,5,5))
        @test !(matrix isa Matrix)
        @test BandedMatrix(matrix) isa BandedMatrix{Int64, Matrix{Int64}}
        @test BandedMatrix(matrix).data isa Matrix{Int64}
        @test BandedMatrix{Int64, typeof(matrix)}(matrix, (1, 1)) isa BandedMatrix{Int64, typeof(matrix)}
        @test BandedMatrix{Int64, typeof(matrix)}(matrix, (1, 1)).data isa typeof(matrix)
        @test_throws UndefRefError BandedMatrix{Vector{Float64}}(undef, (5,5), (1,1))[1,1]

        @testset "parent" begin
            A = zeros(3,5)
            B = _BandedMatrix(A, 5, 1, 1)
            @test parent(B) === A
        end

        @testset "Construction from Diagonal" begin
            D = Diagonal(1:4)
            B1 = BandedMatrix(D)
            @test B1 isa BandedMatrix{Int}
            @test B1 == D
            B2 = BandedMatrix{Float64}(D)
            @test B2 isa BandedMatrix{Float64}
            @test B2 == D
            B3 = BandedMatrix{Float32,Matrix{Float32}}(D)
            @test B3 isa BandedMatrix{Float32,Matrix{Float32}}
            @test B3 == D
        end
    end

    @testset "BandedMatrix arithmetic" begin
        let A = brand(10,12,2,3),B = brand(10,12,3,4)
            @test Matrix(sparse(A)) ≈ Matrix(A)

            @test Matrix(A') ≈ Matrix(A)'
            @test Matrix(transpose(A)) ≈ transpose(Matrix(A))
            @test Matrix((A+im*A)') ≈ (Matrix(A)+im*Matrix(A))'
            @test Matrix(transpose(A+im*A)) ≈ transpose(Matrix(A)+im*Matrix(A))

            @test Matrix(A+B) ≈ (Matrix(A)+Matrix(B))
            @test Matrix(A-B) ≈ (Matrix(A)-Matrix(B))

            @test Matrix(A.*B) ≈ (Matrix(A).*Matrix(B))
        end

        @testset "UniformScaling" begin
            A = brand(10,10,1,2)
            @test A+I isa BandedMatrix
            @test bandwidths(A+I) == (1,2)
            @test Matrix(A+I) ≈ Matrix(A)+I
            @test Matrix(I-A) ≈ I-Matrix(A)


            # number
            @test Matrix(A/2) ≈ Matrix(A)/2
        end
    end

    @testset "BandedMatrix * Vector" begin
        let v=rand(12), w=rand(10)
            for (l,u) in ((2,3), (-2,2), (2,-2), (2,-3))
                A=brand(length(w),length(v),l,u)
                @test A*v ≈ Matrix(A)*v
                # the left-side uses BLAS, while the right doesn't
                @test mul!(ones(size(A,1)), A, v, 1.0, 2.0) ≈ mul!(ones(size(A,1)), A, v, 1, 2)
                @test A'*w ≈ Matrix(A)'*w
                @test mul!(ones(size(A,2)), A', w, 1.0, 2.0) ≈ mul!(ones(size(A,2)), A', w, 1, 2)
                # explicitly test materialize!
                @test materialize!(MulAdd(1.0, A', w, 2.0, ones(size(A,2)))) ≈ materialize!(MulAdd(1, A', w, 2, ones(size(A,2))))
            end
        end

        let A=brand(Float64,5,3,2,2), v=rand(ComplexF64,3), w=rand(ComplexF64,5)
            @test A*v ≈ Matrix(A)*v
            @test mul!(ones(ComplexF64,size(A,1)), A, v, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,1)), A, v, 1, 2)
            @test A'*w ≈ Matrix(A)'*w
            @test mul!(ones(ComplexF64,size(A,2)), A', w, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,2)), A', w, 1, 2)
        end

        let A=brand(ComplexF64,5,3,2,2), v=rand(ComplexF64,3), w=rand(ComplexF64,5)
            @test A*v ≈ Matrix(A)*v
            @test mul!(ones(ComplexF64,size(A,1)), A, v, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,1)), A, v, 1, 2)
            @test A'*w ≈ Matrix(A)'*w
            @test mul!(ones(ComplexF64,size(A,2)), A', w, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,2)), A', w, 1, 2)
        end

        let A=brand(ComplexF64,5,3,2,2), v=rand(Float64,3), w=rand(Float64,5)
            @test A*v ≈ Matrix(A)*v
            @test mul!(ones(ComplexF64,size(A,1)), A, v, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,1)), A, v, 1, 2)
            @test A'*w ≈ Matrix(A)'*w
            @test mul!(ones(ComplexF64,size(A,2)), A', w, 1.0, 2.0) ≈ mul!(ones(ComplexF64,size(A,2)), A', w, 1, 2)
        end

        @testset "empty" begin
            let B=BandedMatrix((0=>ones(0),), (10,0)), v = ones(size(B,2))
                @test B * v == zeros(size(B,1))
            end
            let B=BandedMatrix((0=>ones(0),), (0,10)), v = ones(size(B,2))
                @test B * v == zeros(size(B,1))
            end
        end
    end

    @testset "Banded * Dense" begin
        @testset "big banded * dense" begin
            A=brand(1000,1000,200,300)
            B=rand(1000,1000)
            @test A*B ≈ Matrix(A)*B
            @test B*A ≈ B*Matrix(A)
        end
        # gbmm! not yet implemented
        # @test A'*B ≈ Matrix(A)'*B
        # @test A*B' ≈ Matrix(A)*B'
        # @test A'*B' ≈ Matrix(A)'*B'

        let
            A=brand(1200,1000,200,300); B=rand(1000,1000); C=rand(1200,1200)
            @test A*B ≈ Matrix(A)*B
            @test C*A ≈ C*Matrix(A)
        end
        # gbmm! not yet implemented
        # @test A'*C ≈ Matrix(A)'*C
        # @test A*B' ≈ Matrix(A)*B'
        # @test A'*C' ≈ Matrix(A)'*C'
    end

    @testset "Banded * Banded" begin
        for n in (1,5), ν in (1,5), m in (1,5), Al in (0,1,3), Au in (0,1,3),
                Bl in (0,1,3), Bu in (0,1,3)
            A = brand(n, ν, Al, Au); B = brand(ν, m, Bl, Bu)
            C = brand(ν, n, Al, Bu); D = brand(m, ν, Al, Bu)
            @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
            @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
            @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
            @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
        end

        let A = brand(ComplexF64, 5, 4, 2, 3), B = brand(ComplexF64, 4, 6, 3, 1),
            C = brand(ComplexF64, 4, 5, 1, 1), D = brand(ComplexF64, 6, 4, 0, 3)
            @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
            @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
            @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
            @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
        end

        let A = brand(ComplexF64, 5, 4, 2, 3), B = brand(4, 6, 3, 1), C = brand(4, 5, 1, 1),
                D = brand(ComplexF64, 6, 4, 0, 3)
            @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
            @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
            @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
            @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
        end

        @testset "BigFloat" begin
            let A = brand(5, 5, 1, 2), B = BandedMatrix(Zeros{BigFloat}(5,5),(2,3)), D = rand(5, 5)
                for j = 1:size(B,2), k = colrange(B,j)
                    B[k,j]=randn()
                end

                x = BigFloat[1:size(B,1)...]

                @test Matrix(A)*Matrix(B) ≈ A*B
                @test Matrix(B)*Matrix(A) ≈ B*A
                @test Matrix(B)*x ≈ B*x
                @test Matrix(B*B) ≈ Matrix(B)*Matrix(B)
                @test Matrix(A)*Matrix(D) ≈ A*D
                @test Matrix(D)*Matrix(A) ≈ D*A
            end
        end

        @testset "Negative bands" begin
            for A in (brand(3,4,-1,2),brand(5,4,-1,2),
                        brand(3,4,2,-1),brand(5,4,2,-1))
                b = rand(size(A,2))
                c = rand(size(A,1))
                @test A*b ≈ Matrix(A)*b
                @test A'*c ≈ Matrix(A)'*c
            end

            let C = brand(4, 5, -1, 3), D = rand(4, 4)
                for A in (brand(3,4,1,2),brand(3,4,-1,2),brand(3,4,2,-1)),
                    B in (brand(4,5,1,2),brand(4,5,-1,2),brand(4,5,2,-1))
                    @test A*B ≈ Matrix(A)*Matrix(B)
                    @test B*C' ≈ Matrix(B)*Matrix(C)'
                    @test B'*C ≈ Matrix(B)'*Matrix(C)
                    @test B'*A' ≈ Matrix(B)'*Matrix(A)'
                end

                for A in (brand(5,4,-1,2),brand(5,4,2,-1),brand(3,4,-1,2),brand(3,4,2,-1))
                    @test A*D ≈ Matrix(A)*Matrix(D)
                end

                for B in (brand(4,3,-1,2),brand(4,3,2,-1),brand(4,5,-1,2),brand(4,5,2,-1))
                    @test D*B ≈ Matrix(D)*Matrix(B)
                end
            end
        end

        @testset "zero" begin
            let b = rand(4)
                for A in (brand(3,4,-1,0),brand(3,4,0,-1),brand(3,4,-1,-1)),
                    B in (brand(4,3,1,2),brand(4,3,-1,0),brand(4,3,-1,-1))
                    @test Matrix(A) == zeros(3,4)
                    @test A*B == zeros(3,3)
                    @test A*b == zeros(3)
                end
            end
        end

        @testset "errors in collect" begin
           let B = brand(10,10,0,4)
               @test B*[collect(1.0:10) collect(1.0:10)] ≈ Matrix(B)*[collect(1.0:10) collect(1.0:10)]
           end
        end
    end

    @testset "BandedMatrix interface" begin
        # check that col/rowstop is ≥ 0
        let A = brand(3,4,-2,2)
            @test BandedMatrices.colstop(A, 1) == BandedMatrices.colstop(A, 2) == 0
            @test BandedMatrices.colstop(A, 3) == 1
        end

        let A = brand(3,4,2,-2)
            @test BandedMatrices.rowstop(A, 1) == BandedMatrices.rowstop(A, 2) == 0
            @test BandedMatrices.rowstop(A, 3) == 1
        end

        # Test fill!
        let B = brand(10,10,1,4)
            @test_throws BandError fill!(B, 1.0)
            @test_throws BandError fill!(B, 1)
            fill!(B, 0)
            @test Matrix(B) == zeros(10,10)
        end
    end


    @testset "Conversions" begin
        @testset "banded -> some matrix" begin
            banded = brand(Int32, 10,12,2,3)

            matrix = Matrix(banded)
            @test matrix isa Matrix{Int32}
            for i in 1:length(matrix)
                @test matrix[i] == banded[i]
            end

            matrix = convert(Matrix, banded)
            @test matrix isa Matrix{Int32}
            for i in 1:length(matrix)
                @test matrix[i] == banded[i]
            end

            matrix = convert(Matrix{Int64}, banded)
            @test matrix isa Matrix{Int64}
            for i in 1:length(matrix)
                @test matrix[i] == banded[i]
            end

            matrix = convert(BandedMatrix, banded)
            @test matrix === banded

            matrix = convert(BandedMatrix{Int64}, banded)
            @test matrix isa BandedMatrix{Int64, Matrix{Int64}}
            for i in 1:length(matrix)
                @test matrix[i] == banded[i]
            end
        end

        @testset "some matrix -> banded" begin
            matrix = rand(Int32, 10, 12)

            types = Dict(BandedMatrix => Matrix{Int32},
                         BandedMatrix{Int64} => Matrix{Int64},
                         BandedMatrix{Int64, MyMatrix{Int64}} => MyMatrix{Int64})
            @testset "Matrix to $Final via $Step" for (Step, Final) in types
                banded = @inferred convert(Step, matrix)
                @test banded isa BandedMatrix{eltype(Final), Final}
                @test banded == matrix
                @test @inferred(convert(BandedMatrix, banded)) === banded
            end

            # Note: @inferred convert(MyMatrix, matrix) throws
            banded = convert(BandedMatrix{Int64, MyMatrix}, matrix)
            @test banded isa BandedMatrix{Int64, MyMatrix{Int64}}
            @test banded == matrix

            banded = convert(BandedMatrix{<:, MyMatrix}, matrix)
            @test banded isa BandedMatrix{Int32, MyMatrix{Int32}}
            @test banded == matrix

            jlarray = convert(MyMatrix, rand(Int32, 10, 12))
            types = Dict(BandedMatrix => MyMatrix{Int32},
                         BandedMatrix{Int64} => MyMatrix{Int64},
                         BandedMatrix{Int64, Matrix} => Matrix{Int64},
                         BandedMatrix{<:, Matrix} => Matrix{Int32},
                         BandedMatrix{<:, Matrix{Int32}} => Matrix{Int32},
                         BandedMatrix{<:, Matrix{Int64}} => Matrix{Int64},
                         BandedMatrix{Int64, Matrix{Int64}} => Matrix{Int64})
            @testset "MyMatrix to $Final via $Step" for (Step, Final) in types
                banded = @inferred convert(Step, jlarray)
                @test banded isa BandedMatrix{eltype(Final), Final}
                @test banded == jlarray
            end

            banded[5, 5] = typemax(Int32)
            @test_throws InexactError convert(BandedMatrix{Int16}, matrix)
            @test_throws InexactError convert(BandedMatrix{Int16, MyMatrix}, matrix)
            @test_throws InexactError convert(BandedMatrix{Int16, MyMatrix{Int16}}, matrix)

            banded = @inferred BandedMatrix{eltype(jlarray), typeof(jlarray)}(jlarray, (1, 1))
            types = [BandedMatrix, BandedMatrix{Int32},
                     BandedMatrix{<:, MyMatrix},
                     BandedMatrix{<:, MyMatrix{Int32}}]
            @testset "no-op banded to banded via $Step" for Step in types
                @test @inferred(convert(Step, banded)) === banded
            end

            types = Dict(BandedMatrix{Int64} => MyMatrix{Int64},
                         BandedMatrix{Int64, Matrix} => Matrix{Int64},
                         BandedMatrix{<:, MyMatrix{Int64}} => MyMatrix{Int64},
                         BandedMatrix{<:, Matrix} => Matrix{Int32},
                         BandedMatrix{<:, Matrix{Int32}} => Matrix{Int32},
                         BandedMatrix{<:, Matrix{Int64}} => Matrix{Int64},
                         BandedMatrix{Int64, Matrix{Int64}} => Matrix{Int64})
            @testset "banded to banded $Final via $Step" for (Step, Final) in types
                newbanded = @inferred convert(Step, banded)
                @test newbanded isa BandedMatrix{eltype(Final), Final}
                @test newbanded == banded
            end

            @testset "banded to banded" begin
                A = BandedMatrix{Int}(undef, (5,5), (1,1)); A.data[:] .= 1:length(A.data)
                @test convert(BandedMatrix, A) === convert(BandedMatrix{Int}, A) ===
                    convert(BandedMatrix{Int,Matrix{Int}}, A) === convert(BandedMatrix{Int,Matrix{Int},Base.OneTo{Int}}, A) === A
                @test convert(BandedMatrix{Float64}, A) ==
                    convert(BandedMatrix{Float64,Matrix{Float64}}, A) == convert(BandedMatrix{Float64,Matrix{Float64},Base.OneTo{Int}}, A) == A
            end
        end

        @testset "similar" begin
            banded = brand(Int32, 10, 12, 1, 2)
            @test @inferred(similar(banded)) isa BandedMatrix{Int32, Matrix{Int32}}
            @test size(similar(banded)) == size(banded)
            @test similar(banded).l == banded.l
            @test similar(banded).u == banded.u
            @test @inferred(similar(banded,(5,5))) isa BandedMatrix{Int32, Matrix{Int32}}
            @test @inferred(similar(banded,5,5)) isa BandedMatrix{Int32, Matrix{Int32}}
            @test @inferred(similar(banded,5,5,1,1)) isa BandedMatrix{Int32, Matrix{Int32}}


            banded = convert(BandedMatrix{<:, MyMatrix}, brand(Int32, 10, 12, 1, 2))
            @test banded isa BandedMatrix{Int32, MyMatrix{Int32}}
            @test @inferred(similar(banded)) isa BandedMatrix{Int32, MyMatrix{Int32}}
            @test size(similar(banded)) == size(banded)
            @test similar(banded).l == banded.l
            @test similar(banded).u == banded.u

            banded = convert(BandedMatrix{<:, MyMatrix}, brand(Int32, 10, 12, 1, 2))
            @test @inferred(similar(banded, Int64)) isa BandedMatrix{Int64, MyMatrix{Int64}}
            @test size(similar(banded, Int64)) == size(banded)
            @test similar(banded, Int64).l == banded.l
            @test similar(banded, Int64).u == banded.u

            bview = @view banded[1:5, 1:5]
            expected = convert(BandedMatrix, bview)
            @test @inferred(similar(bview)) isa BandedMatrix{Int32, MyMatrix{Int32}}
            @test @inferred(similar(bview, Int64)) isa BandedMatrix{Int64, MyMatrix{Int64}}
            @test size(similar(bview, Int64)) == size(expected)
            @test bandwidth(similar(bview, Int64), 1) == bandwidth(expected, 1)
            @test bandwidth(similar(bview, Int64), 2) == bandwidth(expected, 2)

            bview = @view banded[1:5, 6:8]
            expected = convert(BandedMatrix, bview)
            @test size(similar(bview, Int64)) == size(expected)
            @test bandwidth(similar(bview, Int64), 1) == bandwidth(expected, 1)
            @test bandwidth(similar(bview, Int64), 2) == bandwidth(expected, 2)

            bview = @view banded[6:8, 1:5]
            expected = convert(BandedMatrix, bview)
            @test size(similar(bview, Int64)) == size(expected)
            @test bandwidth(similar(bview, Int64), 1) == bandwidth(expected, 1)
            @test bandwidth(similar(bview, Int64), 2) == bandwidth(expected, 2)
        end

        @testset "banded -> sparse" begin
            B = brand(5,5,1,1)
            @test Matrix(sparse(B)) == Matrix(B)
        end
    end

    @testset "real-imag" begin
        B = brand(ComplexF64,5,5,1,1)
        @test real(B) isa BandedMatrix
        @test imag(B) isa BandedMatrix
        @test bandwidths(real(B)) == bandwidths(imag(B)) == bandwidths(B)
        @test real(B) + im*imag(B) == B
    end

    @testset "induced norm" begin
        for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
            A = brand(T, 12, 10, 2, 3)
            B = Matrix(A)
    	    @test opnorm(A) ≈ opnorm(B)
    	    @test cond(A) ≈ cond(B)
        end
    end

    @testset "triu/tril" begin
        n = 20; A = brand(n,n,2,3); B = Matrix(A);
        @test triu(A) == triu(B)
        @test tril(A) == tril(B)
        for j = -5:5
            @test triu(A,j) == triu(B,j)
            @test tril(A,j) == tril(B,j)
        end
    end

    @testset "BandedMatrix{Any}" begin
        A = BandedMatrix{Any}(undef, (10,10), (1,1))
        @test A[1,3] === nothing
    end

    @testset "negative bands + (#164)" begin
        A,B = BandedMatrix(-3 => 1:5) , BandedMatrix(3 => 1:5)
        @test A + B == A .+ B == B + A == B .+ A == Matrix(A) + Matrix(B)
        @test A - B == A .- B == Matrix(A) - Matrix(B)
        @test B - A == B .- A == Matrix(B) - Matrix(A)
    end

    @testset "summary" begin
        A = BandedMatrix{Float64}(undef, (5, 5), (1,2))
        @test summary(A) == "5×5 BandedMatrix{Float64} with bandwidths (1, 2)"
        A = _BandedMatrix(Fill(1,1,5), 4, 1, -1)
        @test summary(A) == "4×5 BandedMatrix{$Int} with bandwidths (1, -1) with data 1×5 Fill{$Int}"
        A = _BandedMatrix(Fill(1,1,5), Base.Slice(1:4), 1, -1)
        @test summary(A) == "4×5 BandedMatrix{$Int} with bandwidths (1, -1) with data 1×5 Fill{$Int} with indices 1:4×Base.OneTo(5)"
    end

    @testset "setindex" begin
        @testset "setindex! with ranges (#348)" begin
            n = 10;
            X1 = brand(n,n,1,1)
            B = BandedMatrix(Zeros(2n,2n), (3,3))
            B[1:2:end,1:2:end] = X1
            A = zeros(2n,2n)
            A[1:2:end,1:2:end] = X1
            @test A == B
        end
        @testset "vector - integer - colon" begin
            B = BandedMatrix(0=>1:4, 1=>5:7)
            B[1, :] = [10, 20, 0, 0]
            @test B[1, 1:2] == [10,20]
            B[3, :] = [0, 0, -10, -20]
            @test B[3, 3:4] == [-10, -20]
        end
    end


    @testset "copy band to offset vector" begin
        B = BandedMatrix(2=>2:3)
        # test that a BandedMatrix and a Matrix behave identically
        p = zeros(3)
        v = view(p, Base.IdentityUnitRange(2:3))
        for M in (B, Matrix(B))
            p .= 0
            copyto!(v, diag(M, 2))
            @test p[1] == 0
            @test @view(p[2:3]) == 2:3
            copyto!(v, diag(M, 10))
            @test p[1] == 0
            @test @view(p[2:3]) == 2:3
        end
    end

    if isdefined(LinearAlgebra, :copymutable_oftype)
        @testset "copymutable_oftype" begin
            B = _BandedMatrix((2:5)', 4, -2, 2)
            @test LinearAlgebra.copymutable_oftype(B, Float64) == B
            @test LinearAlgebra.copymutable_oftype(B, Float64) isa BandedMatrix{Float64}
            @test LinearAlgebra.copymutable_oftype(B', Float64) == B'
            @test LinearAlgebra.copymutable_oftype(B', Float64) isa Adjoint{Float64,<:BandedMatrix{Float64}}
            @test LinearAlgebra.copymutable_oftype(transpose(B), Float64) == transpose(B)
            @test LinearAlgebra.copymutable_oftype(transpose(B), Float64) isa Transpose{Float64,<:BandedMatrix{Float64}}

            @test LinearAlgebra.copymutable_oftype(UpperTriangular(B), Float64) == B
        end
    end

    @testset "sparse" begin
        B = BandedMatrix(1 => [1:4;])
        B2 = copy(B)
        S = sparse(B) .* 2
        copyto!(B, S)
        @test B == 2B2
        B .= 0
        copyto!(view(B, :, :), S)
        @test B == 2B2
    end

    @testset "offset views" begin
        B = BandedMatrix(0=>1:4)
        A = view(B, Base.IdentityUnitRange(2:4), 1:4)
        @test !any(in(axes(A,1)), BandedMatrices.colrange(A, 1))
        @test BandedMatrices.colrange(A, 2) == 2:2
        @test BandedMatrices.colrange(A, 3) == 3:3
        @test BandedMatrices.colrange(A, 4) == 4:4
    end
end

end # module
