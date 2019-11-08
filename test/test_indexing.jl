using BandedMatrices, LinearAlgebra, ArrayLayouts, Test

import BandedMatrices: rowstart, rowstop, colstart, colstop,
                       rowlength, collength, diaglength, BandedColumns,
                       rowsupport, colsupport

@testset "Indexing" begin
    @testset "BandedMatrix Indexing" begin
        let
            for n in (10,50), m in (12,50), Al in (0,1,2,30), Au in (0,1,2,30)
                A=brand(n,m,Al,Au)
                kr,jr=3:10,5:12
                @test Matrix(A[kr,jr]) ≈ Matrix(A)[kr,jr]
            end
        end
    end

    @testset "rowstart/rowstop" begin
        A = BandedMatrix(Ones(7, 5), (1, 2))
        # 1.0  1.0  1.0  0.0  0.0
        # 1.0  1.0  1.0  1.0  0.0
        # 0.0  1.0  1.0  1.0  1.0
        # 0.0  0.0  1.0  1.0  1.0
        # 0.0  0.0  0.0  1.0  1.0
        # 0.0  0.0  0.0  0.0  1.0
        # 0.0  0.0  0.0  0.0  0.0
        @test (rowstart(A, 1), rowstop(A, 1), rowlength(A, 1)) == (1, 3, 3)
        @test (rowstart(A, 2), rowstop(A, 2), rowlength(A, 2)) == (1, 4, 4)
        @test (rowstart(A, 3), rowstop(A, 3), rowlength(A, 3)) == (2, 5, 4)
        @test (rowstart(A, 4), rowstop(A, 4), rowlength(A, 4)) == (3, 5, 3)
        @test (rowstart(A, 5), rowstop(A, 5), rowlength(A, 5)) == (4, 5, 2)
        @test (rowstart(A, 6), rowstop(A, 6), rowlength(A, 6)) == (5, 5, 1)
        @test (rowstart(A, 7), rowstop(A, 7), rowlength(A, 7)) == (6, 5, 0) # zero length
        @test (colstart(A, 1), colstop(A, 1), collength(A, 1)) == (1, 2, 2)
        @test (colstart(A, 2), colstop(A, 2), collength(A, 2)) == (1, 3, 3)
        @test (colstart(A, 3), colstop(A, 3), collength(A, 3)) == (1, 4, 4)
        @test (colstart(A, 4), colstop(A, 4), collength(A, 4)) == (2, 5, 4)
        @test (colstart(A, 5), colstop(A, 5), collength(A, 5)) == (3, 6, 4)

        A = BandedMatrix(Ones(3, 6), (1, 2))
        # 1.0  1.0  1.0  0.0  0.0  0.0
        # 1.0  1.0  1.0  1.0  0.0  0.0
        # 0.0  1.0  1.0  1.0  1.0  0.0
        @test (rowstart(A, 1), rowstop(A, 1), rowlength(A, 1)) == (1, 3, 3)
        @test (rowstart(A, 2), rowstop(A, 2), rowlength(A, 2)) == (1, 4, 4)
        @test (rowstart(A, 3), rowstop(A, 3), rowlength(A, 3)) == (2, 5, 4)
        @test (colstart(A, 1), colstop(A, 1), collength(A, 1)) == (1, 2, 2)
        @test (colstart(A, 2), colstop(A, 2), collength(A, 2)) == (1, 3, 3)
        @test (colstart(A, 3), colstop(A, 3), collength(A, 3)) == (1, 3, 3)
        @test (colstart(A, 4), colstop(A, 4), collength(A, 4)) == (2, 3, 2)
        @test (colstart(A, 5), colstop(A, 5), collength(A, 5)) == (3, 3, 1)
        @test (colstart(A, 6), colstop(A, 6), collength(A, 6)) == (4, 3, 0) # zero length

        A = BandedMatrix(Ones(3, 4), (-1, 2))
        # 0.0  1.0  1.0  0.0
        # 0.0  0.0  1.0  1.0
        # 0.0  0.0  0.0  1.0
        @test rowrange(A,1) == rowsupport(A,1) == 2:3
        @test rowrange(A,2) == rowsupport(A,2) == 3:4
        @test rowrange(A,3) == rowsupport(A,3) == 4:4
        @test colrange(A,1) == colsupport(A,1) == 1:0
        @test colrange(A,2) == colsupport(A,2) == 1:1
        @test colrange(A,3) == colsupport(A,3) == 1:2
        @test colrange(A,4) == colsupport(A,4) == 2:3
    end

    @testset "test length of diagonal" begin
        A = BandedMatrix(Ones(4, 6), (2, 3))
        # 4x6 BandedMatrices.BandedMatrix{Float64}:
        #  1.0  1.0  1.0  1.0
        #  1.0  1.0  1.0  1.0  1.0
        #  1.0  1.0  1.0  1.0  1.0  1.0
        #       1.0  1.0  1.0  1.0  1.0
        @test diaglength(A, -2) == 2
        @test diaglength(A, -1) == 3
        @test diaglength(A,  0) == 4
        @test diaglength(A,  1) == 4
        @test diaglength(A,  2) == 4
        @test diaglength(A,  3) == 3

        A = BandedMatrix(Ones(6, 5), (2, 1))
        # 6x5 BandedMatrices.BandedMatrix{Float64}:
        #  1.0  1.0
        #  1.0  1.0  1.0
        #  1.0  1.0  1.0  1.0
        #       1.0  1.0  1.0  1.0
        #            1.0  1.0  1.0
        #                 1.0  1.0
        @test diaglength(A, -2) == 4
        @test diaglength(A, -1) == 5
        @test diaglength(A,  0) == 5
        @test diaglength(A,  1) == 4

        A = BandedMatrix(Ones(4, 4), (1, 1))
        # 4x4 BandedMatrices.BandedMatrix{Float64}:
        #  1.0  1.0
        #  1.0  1.0  1.0
        #       1.0  1.0  1.0
        #            1.0  1.0
        @test diaglength(A, -1) == 3
        @test diaglength(A,  0) == 4
        @test diaglength(A,  1) == 3

        # test with Band type
        @test diaglength(A,  band(1)) == 3
    end

    @testset "_firstdiagrow/_firstdiagcol" begin
        A = BandedMatrix(Ones(6, 5), (2, 1))
        # 6x5 BandedMatrices.BandedMatrix{Float64}:
        #  1.0  1.0
        #  1.0  1.0  1.0
        #  1.0  1.0  1.0  1.0
        #       1.0  1.0  1.0  1.0
        #            1.0  1.0  1.0
        #                 1.0  1.0
        @test BandedMatrices._firstdiagrow(A, 1) == 2
        @test BandedMatrices._firstdiagrow(A, 2) == 2
        @test BandedMatrices._firstdiagrow(A, 3) == 2
        @test BandedMatrices._firstdiagrow(A, 4) == -3
        @test BandedMatrices._firstdiagrow(A, 5) == -3
        @test BandedMatrices._firstdiagrow(A, 6) == -3

        @test BandedMatrices._firstdiagcol(A, 1) == -3
        @test BandedMatrices._firstdiagcol(A, 2) == -3
        @test BandedMatrices._firstdiagcol(A, 3) == 2
        @test BandedMatrices._firstdiagcol(A, 4) == 2
        @test BandedMatrices._firstdiagcol(A, 5) == 2
    end


    @testset "scalar - integer - integer" begin
        a = BandedMatrix(Ones(5, 5), (1, 1))
        # 1.0  1.0  0.0  0.0  0.0
        # 1.0  1.0  1.0  0.0  0.0
        # 0.0  1.0  1.0  1.0  0.0
        # 0.0  0.0  1.0  1.0  1.0

        # in band
        a[1, 1] = 2
        @test a[1, 1] == 2

        # out of band
        @test_throws BandError a[1, 3] = 1
        @test_throws BandError a[3, 1] = 1

        # out of range
        @test_throws BoundsError a[0, 0] = 1
        @test_throws BoundsError a[0, 1] = 1
        @test_throws BoundsError a[1, 0] = 1
        @test_throws BoundsError a[5, 6] = 1
        @test_throws BoundsError a[6, 5] = 1
        @test_throws BoundsError a[6, 6] = 1

        a = BandedMatrix(Ones(5, 5), (-1, 1))
        a[1, 2] = 2
        @test a[1, 2] == 2
        @test_throws BandError a[1, 1] = 1
    end


    @testset "indexing along a column" begin
        @testset "scalar - BandRange/Colon - integer" begin
            a = BandedMatrix(Ones(5, 5), (1, 1))
            # 1.0  1.0  0.0  0.0  0.0
            # 1.0  1.0  1.0  0.0  0.0
            # 0.0  1.0  1.0  1.0  0.0
            # 0.0  0.0  1.0  1.0  1.0
            # 0.0  0.0  0.0  1.0  1.0

            # in band
            a[BandRange, 1] .= 2
            a[BandRange, 2] .= 3
            a[BandRange, 3] .= 4
            a[BandRange, 4] .= 5
            a[BandRange, 5] .= 6
            @test a ==  [2  3  0  0  0;
                         2  3  4  0  0;
                         0  3  4  5  0;
                         0  0  4  5  6;
                         0  0  0  5  6]

            @test a[BandRange, 1] == [2, 2]
            @test a[BandRange, 2] == [3, 3, 3]
            @test a[BandRange, 3] == [4, 4, 4]
            @test a[BandRange, 4] == [5, 5, 5]
            @test a[BandRange, 5] == [6, 6]

            @test_throws BoundsError a[:, 0] .= 1
            @test_throws BandError a[:, 1] .= 1
            @test_throws BoundsError a[BandRange, 0] .= 1
            @test_throws BoundsError a[BandRange, 6] .= 1

            a = BandedMatrix(Ones(3, 5), (-1, 2))
            @test isempty(a[BandRange,1])
            a[BandRange,2] = [1]
            a[BandRange,3] = [2, 2]
            a[BandRange,4] = [3, 3]
            a[BandRange,5] = [4]

            @test a ==   [0 1 2 0 0;
                          0 0 2 3 0;
                          0 0 0 3 4]
        end

        @testset "vector - BandRange/Colon - integer" begin
            a = BandedMatrix(Ones{Int}(5, 7), (2, 1))
            # 5x7 BandedMatrices.BandedMatrix{Float64}:
            #  1.0  1.0    0    0    0    0   0   0
            #  1.0  1.0  1.0    0    0    0   0   0
            #  1.0  1.0  1.0  1.0    0    0   0   0
            #    0  1.0  1.0  1.0  1.0    0   0   0
            #    0    0  1.0  1.0  1.0  1.0   0   0

            # in band
            a[BandRange, 1] = [1,   2,  3]
            a[BandRange, 2] = [4,   5,  6,  7]
            a[BandRange, 3] = [8,   9, 10, 11]
            a[BandRange, 4] = [12, 13, 14]
            a[BandRange, 5] = [15, 16]
            a[BandRange, 6] = [17]

            @test a == [ 1  4   0   0   0   0   0;
                         2  5   8   0   0   0   0;
                         3  6   9  12   0   0   0;
                         0  7  10  13  15   0   0;
                         0  0  11  14  16  17   0]

            @test a[BandRange, 1] == [1,   2,  3]
            @test a[BandRange, 2] == [4,   5,  6,  7]
            @test a[BandRange, 3] == [8,   9, 10, 11]
            @test a[BandRange, 4] == [12, 13, 14]
            @test a[BandRange, 5] == [15, 16]
            @test a[BandRange, 6] == [17]

            @test_throws BoundsError a[:, 0] = [1, 2, 3]
            @test_throws DimensionMismatch a[:, 1] = [1, 2, 3]
            @test_throws BoundsError a[BandRange, 0] = [1, 2, 3]
            @test_throws BoundsError a[BandRange, 8] = [1, 2, 3]
            @test_throws DimensionMismatch a[BandRange, 1] = [1, 2]
        end


        # scalar - range - integer
        let
            a = BandedMatrix(Ones(3, 4), (2, 1))
            # 1.0  1.0  0.0  0.0
            # 1.0  1.0  1.0  0.0
            # 1.0  1.0  1.0  1.0

            # Matrix column span
            a[1:3, 1] .= 1
            a[1:3, 2] .= 2
            a[2:3, 3] .= 3
            a[3:3, 4] .= 4
            @test a == [ 1  2  0  0;
                         1  2  3  0;
                         1  2  3  4]

            # partial span
            a[1:1, 1] .= 3
            a[2:3, 2] .= 4
            a[3:3, 3] .= 5
            @test a == [ 3  2  0  0;
                         1  4  3  0;
                         1  4  5  4]

             a[:, 1] = [1, 1, 1]
             a[:, 2] = [2, 2, 2]
             a[:, 3] = [0, 3, 3]
             a[:, 4] = [0, 0, 4]

             @test a == [ 1  2  0  0;
                          1  2  3  0;
                          1  2  3  4]

            # wrong range input
            @test_throws BoundsError   a[0:1, 1] .= 3
            @test_throws BoundsError   a[1:4, 1] .= 3
            @test_throws BandError a[2:3, 4] .= 3
            @test_throws BoundsError   a[3:4, 4] .= 3
        end

        # vector - range - integer
        let
            a = BandedMatrix(Ones(5, 4), (1, 2))
            # 1.0  1.0  1.0  0.0
            # 1.0  1.0  1.0  1.0
            # 0.0  1.0  1.0  1.0
            # 0.0  0.0  1.0  1.0
            # 0.0  0.0  0.0  1.0
            # Matrix column span
            a[1:2, 1] = 1:2
            a[1:3, 2] = 3:5
            a[1:4, 3] = 6:9
            a[2:5, 4] = 10:13
            @test a == [1  3  6   0;
                        2  4  7  10;
                        0  5  8  11;
                        0  0  9  12;
                        0  0  0  13]

            # partial span
            a[1:1, 1] = [2]
            a[2:3, 2] = 1:2
            a[3:4, 3] = 3:4
            a[4:5, 4] = 3:4
            @test a == [2  3  6   0;
                        2  1  7  10;
                        0  2  3  11;
                        0  0  4   3;
                        0  0  0   4]

            # wrong range input
            @test_throws BoundsError a[1:2, 0] = 1:2
            @test_throws BoundsError a[0:1, 1] = 1:2
            @test_throws BoundsError a[1:8, 1] = 1:8
            @test_throws BoundsError a[2:6, 4] = 2:6
            @test_throws BoundsError a[3:4, 5] = 3:4

            a[:, 1] = [1,1,0,0,0]
            a[:, 2] = [2,2,2,0,0]
            a[:, 3] = [3,3,3,3,0]
            a[:, 4] = [0,4,4,4,4]

            @test a == [1  2  3   0;
                        1  2  3   4;
                        0  2  3   4;
                        0  0  3   4;
                        0  0  0   4]
        end
    end


    @testset "indexing along a row" begin
        # scalar - integer - BandRange/colon
        let
            a = BandedMatrix(Ones(5, 5), (1, 1))
            # 1.0  1.0  0.0  0.0  0.0
            # 1.0  1.0  1.0  0.0  0.0
            # 0.0  1.0  1.0  1.0  0.0
            # 0.0  0.0  1.0  1.0  1.0
            # 0.0  0.0  0.0  1.0  1.0

            # in band
            a[1, BandRange] .= 2
            a[2, BandRange] .= 3
            a[3, BandRange] .= 4
            a[4, BandRange] .= 5
            a[5, BandRange] .= 6

            @test a == [2  2  0  0  0;
                        3  3  3  0  0;
                        0  4  4  4  0;
                        0  0  5  5  5;
                        0  0  0  6  6]


            @test_throws BoundsError a[0, :] .= 1
            @test_throws BandError a[1, :] .= 1
            @test_throws BoundsError a[0, BandRange] .= 1
            @test_throws BoundsError a[6, BandRange] .= 1

            a[1, :] .= 0
            @test vec(a[1, :]) == [0, 0, 0, 0, 0]
        end



        # vector - integer - BandRange/colon
        let
            a = BandedMatrix(Ones(7, 5), (1, 2))
            # 7x5 BandedMatrices.BandedMatrix{Float64}:
            # 1.0  1.0  1.0  0.0  0.0
            # 1.0  1.0  1.0  1.0  0.0
            # 0.0  1.0  1.0  1.0  1.0
            # 0.0  0.0  1.0  1.0  1.0
            # 0.0  0.0  0.0  1.0  1.0
            # 0.0  0.0  0.0  0.0  1.0
            # 0.0  0.0  0.0  0.0  0.0

            # in band
            a[1, BandRange] = [1,   2,  3]
            a[2, BandRange] = [4,   5,  6,  7]
            a[3, BandRange] = [8,   9, 10, 11]
            a[4, BandRange] = [12, 13, 14]
            a[5, BandRange] = [15, 16]
            a[6, BandRange] = [17]

            @test a ==  [1  2   3   0   0;
                         4  5   6   7   0;
                         0  8   9  10  11;
                         0  0  12  13  14;
                         0  0   0  15  16;
                         0  0   0   0  17;
                         0  0   0   0   0]

            @test_throws BoundsError a[0, :] = [1, 2, 3]
            @test_throws DimensionMismatch a[1, :] = [1, 2, 3]
            @test_throws BoundsError a[0, BandRange] = [1, 2, 3]
            @test_throws BoundsError a[8, BandRange] = [1, 2, 3]
            @test_throws DimensionMismatch a[1, BandRange] = [1, 2]
            @test_throws DimensionMismatch a[7, BandRange] = [1, 2, 3]

            a[7, BandRange] .= 1
            @test a ==  [1  2   3   0   0;
                         4  5   6   7   0;
                         0  8   9  10  11;
                         0  0  12  13  14;
                         0  0   0  15  16;
                         0  0   0   0  17;
                         0  0   0   0   0]
        end


        # scalar - integer - range
        let
            a = BandedMatrix(Ones(7, 5), (1, 2))
            # 7x5 BandedMatrices.BandedMatrix{Float64}:
            # 1.0  1.0  1.0  0.0  0.0
            # 1.0  1.0  1.0  1.0  0.0
            # 0.0  1.0  1.0  1.0  1.0
            # 0.0  0.0  1.0  1.0  1.0
            # 0.0  0.0  0.0  1.0  1.0
            # 0.0  0.0  0.0  0.0  1.0
            # 0.0  0.0  0.0  0.0  0.0

            # in band
            a[1, 1:3] .= 2
            a[2, 1:4] .= 3
            a[3, 2:5] .= 4
            a[4, 3:5] .= 5
            a[5, 4:5] .= 6
            a[6, 5:5] .= 7

            @test a ==  [2  2  2  0  0;
                         3  3  3  3  0;
                         0  4  4  4  4;
                         0  0  5  5  5;
                         0  0  0  6  6;
                         0  0  0  0  7;
                         0  0  0  0  0]

            a[1, 1:4] .= 0

            @test a ==  [0  0  0  0  0;
                         3  3  3  3  0;
                         0  4  4  4  4;
                         0  0  5  5  5;
                         0  0  0  6  6;
                         0  0  0  0  7;
                         0  0  0  0  0]

            @test_throws BoundsError a[0, 1:3] .= 1
            @test_throws BoundsError a[8, 2:3] .= 1
            @test_throws BoundsError a[1, 0:3] .= 1
            @test_throws BoundsError a[4, 4:6] .= 1

            @test_throws BandError a[1, 1:4] .= 1
        end

        # vector - integer - range
        let
            a = BandedMatrix(Ones(7, 5), (1, 2))
            # 7x5 BandedMatrices.BandedMatrix{Float64}:
            # 1.0  1.0  1.0  0.0  0.0
            # 1.0  1.0  1.0  1.0  0.0
            # 0.0  1.0  1.0  1.0  1.0
            # 0.0  0.0  1.0  1.0  1.0
            # 0.0  0.0  0.0  1.0  1.0
            # 0.0  0.0  0.0  0.0  1.0
            # 0.0  0.0  0.0  0.0  0.0

            # in band
            a[1, 1:3] = [1, 2, 3]
            a[2, 1:4] = [1, 2, 3, 4]
            a[3, 2:5] = [1, 2, 3, 4]
            a[4, 3:5] = [1, 2, 3]
            a[5, 4:5] = [1, 2]
            a[6, 5:5] = [1]

            @test a ==  [1  2  3  0  0;
                         1  2  3  4  0;
                         0  1  2  3  4;
                         0  0  1  2  3;
                         0  0  0  1  2;
                         0  0  0  0  1;
                         0  0  0  0  0]

            @test_throws BoundsError a[0, 1:3] = [1, 2, 3]
            @test_throws BoundsError a[8, 2:3] = [1, 2]
            @test_throws BoundsError a[1, 0:3] = [1, 2, 3, 4]
            @test_throws BoundsError a[4, 4:6] = [2, 3]

            @test_throws BandError a[1, 1:4] = [1, 2, 3, 4]
            @test_throws BoundsError a[7, 6:7] = [1, 2]
            @test_throws DimensionMismatch a[1, 1:3] = [1, 2]
        end
    end

    @testset "indexing along a band" begin
        let
            a = BandedMatrix(Zeros(5, 4), (2, 1))
            # 5x4 BandedMatrices.BandedMatrix{Float64}:
            #  0.0  0.0
            #  0.0  0.0  0.0
            #  0.0  0.0  0.0  0.0
            #       0.0  0.0  0.0
            #            0.0  0.0
            a[band(-2)] .= 5
            a[band(-1)] .= 1
            a[band( 0)] .= 2
            a[band( 1)] .= 3

            @test Matrix(a) == [ 2  3  0  0;
                               1  2  3  0;
                               5  1  2  3;
                               0  5  1  2;
                               0  0  5  1]

            @test_throws BandError a[band(-3)] .= 1

            a[band(-2)] = [4, 4, 4]
            a[band(-1)] = [1, 2, 3, 4]
            a[band( 0)] = [5, 6, 7, 8]
            a[band( 1)] = [9, 10, 11]

            @test Matrix(a) == [ 5  9   0   0;
                               1  6  10   0;
                               4  2   7  11;
                               0  4   3   8;
                               0  0   4   4]

            @test a[band(-2)] == [4, 4, 4]
            @test a[band(-1)] == [1, 2, 3, 4]
            @test a[band( 0)] == [5, 6, 7, 8]
            @test a[band( 1)] == [9, 10, 11]


            @test_throws BandError a[band(-3)] = [1, 2, 3]
        end


        let
            a = BandedMatrix(Zeros(5, 4), (2, 2))
            # 5x4 BandedMatrices.BandedMatrix{Float64}:
            #  0.0  0.0  0.0
            #  0.0  0.0  0.0  0.0
            #  0.0  0.0  0.0  0.0
            #       0.0  0.0  0.0
            #            0.0  0.0
            a[band(-2)] .= 5
            a[band(-1)] .= 1
            a[band( 0)] .= 2
            a[band( 1)] .= 3
            a[band( 2)] .= 4

            @test Matrix(a) == [ 2  3  4  0;
                               1  2  3  4;
                               5  1  2  3;
                               0  5  1  2;
                               0  0  5  1]

            @test_throws BandError a[band(-3)] .= 1

            a[band(-2)] = [4,  4,  4]
            a[band(-1)] = [1,  2,  3, 4]
            a[band( 0)] = [5,  6,  7, 8]
            a[band( 1)] = [9,  10, 11]
            a[band( 2)] = [12, 13]

            @test Matrix(a) == [ 5  9   12  0;
                               1  6  10  13;
                               4  2   7  11;
                               0  4   3   8;
                               0  0   4   4]

            @test a[band(-2)] == [4, 4, 4]
            @test a[band(-1)] == [1, 2, 3, 4]
            @test a[band( 0)] == [5, 6, 7, 8]
            @test a[band( 1)] == [9, 10, 11]
            @test a[band( 2)] == [12, 13]


            @test_throws BandError a[band(-3)] = [1, 2, 3]
        end
    end

    @testset "other special indexing" begin
        @testset "all elements" begin
            a = BandedMatrix(Ones(3, 3), (1, 1))
            a[:] .= 0
            @test a == [0 0 0;
                        0 0 0;
                        0 0 0]

            # all rows/cols
            a.data .= 2
            @test a == [2 2 0;
                        2 2 2;
                        0 2 2]

            a[:, :] = [3 3 0;
                        3 3 3;
                        0 3 3]
            @test a == [3 3 0;
                        3 3 3;
                        0 3 3]

            @test_throws BandError a[:] .= 1
            @test_throws BandError a[:,:] .= 1
            @test_throws BandError a[:,:] = ones(3,3)
        end

        @testset "replace a block in the band" begin
            a = BandedMatrix(Zeros(5, 4), (2, 1))
            # 5x4 BandedMatrices.BandedMatrix{Float64}:
            #  0.0  0.0
            #  0.0  0.0  0.0
            #  0.0  0.0  0.0  0.0
            #       0.0  0.0  0.0
            #            0.0  0.0
            a[1:3, 1:2] .= 2
            a[3:5, 3:4] = [1 2;
                           3 4;
                           5 6]
            @test Matrix(a) == [2  2  0 0;
                              2  2  0 0;
                              2  2  1 2;
                              0  0  3 4;
                              0  0  5 6]

            @test_throws BoundsError a[0:3, 1:2] .= 2
            @test_throws BoundsError a[3:6, 3:4] .= 2
            @test_throws BoundsError a[1:3, 0:2] .= 2
            @test_throws BoundsError a[3:5, 3:5] .= 2
            @test_throws BoundsError a[0:3, 1:2] = [1 2; 3 4]
            @test_throws BoundsError a[3:6, 3:4] = [1 2; 3 4]
            @test_throws BoundsError a[1:3, 0:2] = [1 2; 3 4]
            @test_throws BoundsError a[3:5, 3:5] = [1 2; 3 4]

            @test_throws BandError a[1:3, 1:3] .= 2
            @test_throws BandError a[1:3, 1:3] = rand(3, 3)

            @test_throws DimensionMismatch a[1:3, 1:2] = rand(3, 3)

            s = BandedMatrix(view(a,1:3,1:2))
            @test isa(s,BandedMatrix)
            @test Matrix(s) == a[1:3,1:2]
        end

        @testset "tests bug" begin
            a = BandedMatrix(Zeros(1,1), (3,-1))
            @test a[band(-2)] == Float64[]
        end

        @testset "band views" begin
            for A in (rand(11,10), brand(11,10,2,3), brand(Float32, 11,10,2,3),
                         brand(ComplexF64, 11,10,2,3))
                for k = -5:5
                    V = view(A, band(k))
                    bs = BandedMatrices.parentindices(V)[1] # a bandslice
                    @test bs.indices == diagind(A, k)
                    @test bs.band == Band(k)
                    @test collect(bs) == collect(diagind(A, k))
                    @test Vector{eltype(A)}(V) == collect(V) == A[diagind(A,k)] == A[band(k)]
                    @test Vector{ComplexF64}(V) == Vector{ComplexF64}(A[diagind(A,k)]) ==
                            convert(AbstractVector{ComplexF64}, V) == convert(AbstractArray{ComplexF64}, V)
                    @test V ≡ convert(AbstractArray, V) ≡ convert(AbstractArray{eltype(A)}, V) ≡
                            convert(AbstractArray, V) ≡ convert(AbstractVector, V)
                end
            end
        end
    end

    @testset "Sub-banded views" begin
        A = brand(10,10,2,1)
        for V in (view(A,Base.OneTo(5),Base.OneTo(6)), view(A,1:5,1:6))
            @test MemoryLayout(typeof(V)) == BandedColumns{DenseColumnMajor}()
            @test V isa BandedMatrices.BandedSubBandedMatrix
            @test V[3,3] == BandedMatrices.inbands_getindex(V,3,3) == A[3,3]
            V[3,3] = 2
            @test V[3,3] == BandedMatrices.inbands_getindex(V,3,3) == A[3,3] == 2
            BandedMatrices.inbands_setindex!(V,5,3,3)
            @test V[3,3] == BandedMatrices.inbands_getindex(V,3,3) == A[3,3] == 5
        end

        @test A[Base.OneTo(5),Base.OneTo(6)] isa BandedMatrix
        @test A[2:5,Base.OneTo(6)] isa BandedMatrix
        @test A[2:2:5,Base.OneTo(6)] isa Matrix
    end
end
