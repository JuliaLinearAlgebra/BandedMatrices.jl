using Base.Test
using BandedMatrices
import BandedMatrices: rowstart,
                       rowstop,
                       colstart,
                       colstop,
                       rowlength,
                       collength,
                       diaglength

# rowstart/rowstop business
let
    a = bones(7, 5, 1, 2)
    # 1.0  1.0  1.0  0.0  0.0
    # 1.0  1.0  1.0  1.0  0.0
    # 0.0  1.0  1.0  1.0  1.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0
    # 0.0  0.0  0.0  0.0  1.0
    # 0.0  0.0  0.0  0.0  0.0
    @test (rowstart(a, 1), rowstop(a, 1), rowlength(a, 1)) == (1, 3, 3)
    @test (rowstart(a, 2), rowstop(a, 2), rowlength(a, 2)) == (1, 4, 4)
    @test (rowstart(a, 3), rowstop(a, 3), rowlength(a, 3)) == (2, 5, 4)
    @test (rowstart(a, 4), rowstop(a, 4), rowlength(a, 4)) == (3, 5, 3)
    @test (rowstart(a, 5), rowstop(a, 5), rowlength(a, 5)) == (4, 5, 2)
    @test (rowstart(a, 6), rowstop(a, 6), rowlength(a, 6)) == (5, 5, 1)
    @test (rowstart(a, 7), rowstop(a, 7), rowlength(a, 7)) == (6, 5, 0) # zero length
    @test (colstart(a, 1), colstop(a, 1), collength(a, 1)) == (1, 2, 2)
    @test (colstart(a, 2), colstop(a, 2), collength(a, 2)) == (1, 3, 3)
    @test (colstart(a, 3), colstop(a, 3), collength(a, 3)) == (1, 4, 4)
    @test (colstart(a, 4), colstop(a, 4), collength(a, 4)) == (2, 5, 4)
    @test (colstart(a, 5), colstop(a, 5), collength(a, 5)) == (3, 6, 4)

    a = bones(3, 6, 1, 2)
    # 1.0  1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  1.0  0.0
    @test (rowstart(a, 1), rowstop(a, 1), rowlength(a, 1)) == (1, 3, 3)
    @test (rowstart(a, 2), rowstop(a, 2), rowlength(a, 2)) == (1, 4, 4)
    @test (rowstart(a, 3), rowstop(a, 3), rowlength(a, 3)) == (2, 5, 4)
    @test (colstart(a, 1), colstop(a, 1), collength(a, 1)) == (1, 2, 2)
    @test (colstart(a, 2), colstop(a, 2), collength(a, 2)) == (1, 3, 3)
    @test (colstart(a, 3), colstop(a, 3), collength(a, 3)) == (1, 3, 3)
    @test (colstart(a, 4), colstop(a, 4), collength(a, 4)) == (2, 3, 2)
    @test (colstart(a, 5), colstop(a, 5), collength(a, 5)) == (3, 3, 1)
    @test (colstart(a, 6), colstop(a, 6), collength(a, 6)) == (4, 3, 0) # zero length
end

# test length of diagonal
let
    A = bones(4, 6, 2, 3)
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

    A = bones(6, 5, 2, 1)
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

    A = bones(4, 4, 1, 1)
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

# _firstdiagrow/_firstdiagcol
let
    A = bones(6, 5, 2, 1)
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

# scalar - integer - integer
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0

    # in band
    a[1, 1] = 1
    @test a[1, 1] == 1

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
end

# ~ indexing along a column

# scalar - BandRange/Colon - integer
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0

    # in band
    a[BandRange, 1] = 2
    a[BandRange, 2] = 3
    a[BandRange, 3] = 4
    a[BandRange, 4] = 5
    a[BandRange, 5] = 6
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

    @test_throws BandError a[:, 0] = 1
    @test_throws BandError a[:, 1] = 1
    @test_throws BoundsError a[BandRange, 0] = 1
    @test_throws BoundsError a[BandRange, 6] = 1
end



# vector - BandRange/Colon - integer
let
    a = bones(Int, 5, 7, 2, 1)
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
    a = bones(3, 4, 2, 1)
    # 1.0  1.0  0.0  0.0
    # 1.0  1.0  1.0  0.0
    # 1.0  1.0  1.0  1.0

    # full column span
    a[1:3, 1] = 1
    a[1:3, 2] = 2
    a[2:3, 3] = 3
    a[3:3, 4] = 4
    @test a == [ 1  2  0  0;
                 1  2  3  0;
                 1  2  3  4]

    # partial span
    a[1:1, 1] = 3
    a[2:3, 2] = 4
    a[3:3, 3] = 5
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
    @test_throws BoundsError   a[0:1, 1] = 3
    @test_throws BoundsError   a[1:4, 1] = 3
    @test_throws BandError a[2:3, 4] = 3
    @test_throws BoundsError   a[3:4, 4] = 3
end

# vector - range - integer
let
    a = bones(5, 4, 1, 2)
    # 1.0  1.0  1.0  0.0
    # 1.0  1.0  1.0  1.0
    # 0.0  1.0  1.0  1.0
    # 0.0  0.0  1.0  1.0
    # 0.0  0.0  0.0  1.0
    # full column span
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



# ~ indexing along a row

# scalar - integer - BandRange/colon
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0

    # in band
    a[1, BandRange] = 2
    a[2, BandRange] = 3
    a[3, BandRange] = 4
    a[4, BandRange] = 5
    a[5, BandRange] = 6

    @test a == [2  2  0  0  0;
                3  3  3  0  0;
                0  4  4  4  0;
                0  0  5  5  5;
                0  0  0  6  6]


    @test_throws BandError a[0, :] = 1
    @test_throws BandError a[1, :] = 1
    @test_throws BoundsError a[0, BandRange] = 1
    @test_throws BoundsError a[6, BandRange] = 1

    a[1, :] = 0
    @test vec(a[1, :]) == [0, 0, 0, 0, 0]
end



# vector - integer - BandRange/colon
let
    a = bones(7, 5, 1, 2)
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
    @test_throws BoundsError a[7, BandRange] = [1, 2, 3]
    @test_throws BoundsError a[7, BandRange] = 1
end


# scalar - integer - range
let
    a = bones(7, 5, 1, 2)
    # 7x5 BandedMatrices.BandedMatrix{Float64}:
    # 1.0  1.0  1.0  0.0  0.0
    # 1.0  1.0  1.0  1.0  0.0
    # 0.0  1.0  1.0  1.0  1.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0
    # 0.0  0.0  0.0  0.0  1.0
    # 0.0  0.0  0.0  0.0  0.0

    # in band
    a[1, 1:3] = 2
    a[2, 1:4] = 3
    a[3, 2:5] = 4
    a[4, 3:5] = 5
    a[5, 4:5] = 6
    a[6, 5:5] = 7

    @test a ==  [2  2  2  0  0;
                 3  3  3  3  0;
                 0  4  4  4  4;
                 0  0  5  5  5;
                 0  0  0  6  6;
                 0  0  0  0  7;
                 0  0  0  0  0]

    a[1, 1:4] = 0

    @test a ==  [0  0  0  0  0;
                 3  3  3  3  0;
                 0  4  4  4  4;
                 0  0  5  5  5;
                 0  0  0  6  6;
                 0  0  0  0  7;
                 0  0  0  0  0]

    @test_throws BoundsError a[0, 1:3] = 1
    @test_throws BoundsError a[8, 2:3] = 1
    @test_throws BoundsError a[1, 0:3] = 1
    @test_throws BoundsError a[4, 4:6] = 1

    @test_throws BandError a[1, 1:4] = 1
end

# vector - integer - range
let
    a = bones(7, 5, 1, 2)
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

# indexing along a band
let
    a = bzeros(5, 4, 2, 1)
    # 5x4 BandedMatrices.BandedMatrix{Float64}:
    #  0.0  0.0
    #  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0
    #       0.0  0.0  0.0
    #            0.0  0.0
    a[band(-2)] = 5
    a[band(-1)] = 1
    a[band( 0)] = 2
    a[band( 1)] = 3

    @test full(a) == [ 2  3  0  0;
                       1  2  3  0;
                       5  1  2  3;
                       0  5  1  2;
                       0  0  5  1]

    @test_throws BandError a[band(-3)] = 1

    a[band(-2)] = [4, 4, 4]
    a[band(-1)] = [1, 2, 3, 4]
    a[band( 0)] = [5, 6, 7, 8]
    a[band( 1)] = [9, 10, 11]

    @test full(a) == [ 5  9   0   0;
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




# other special methods
let
    # all elements
    a = bones(3, 3, 1, 1)
    a[:] = 0
    @test a == [0 0 0;
                0 0 0;
                0 0 0]

    # all rows/cols
    a[BandRange] = 2
    @test a == [2 2 0;
                2 2 2;
                0 2 2]

    a[:, :] = [3 3 0;
                3 3 3;
                0 3 3]
    @test a == [3 3 0;
                3 3 3;
                0 3 3]

    @test_throws BandError a[:] = 1
    @test_throws BandError a[:,:] = 1
    @test_throws BandError a[:,:] = ones(3,3)
end



# replace a block in the band
let
    a = bzeros(5, 4, 2, 1)
    # 5x4 BandedMatrices.BandedMatrix{Float64}:
    #  0.0  0.0
    #  0.0  0.0  0.0
    #  0.0  0.0  0.0  0.0
    #       0.0  0.0  0.0
    #            0.0  0.0
    a[1:3, 1:2] = 2
    a[3:5, 3:4] = [1 2;
                   3 4;
                   5 6]
    @test full(a) == [2  2  0 0;
                      2  2  0 0;
                      2  2  1 2;
                      0  0  3 4;
                      0  0  5 6]

    @test_throws BoundsError a[0:3, 1:2] = 2
    @test_throws BoundsError a[3:6, 3:4] = 2
    @test_throws BoundsError a[1:3, 0:2] = 2
    @test_throws BoundsError a[3:5, 3:5] = 2
    @test_throws BoundsError a[0:3, 1:2] = [1 2; 3 4]
    @test_throws BoundsError a[3:6, 3:4] = [1 2; 3 4]
    @test_throws BoundsError a[1:3, 0:2] = [1 2; 3 4]
    @test_throws BoundsError a[3:5, 3:5] = [1 2; 3 4]

    @test_throws BandError a[1:3, 1:3] = 2
    @test_throws BandError a[1:3, 1:3] = rand(3, 3)

    @test_throws DimensionMismatch a[1:3, 1:2] = rand(3, 3)
end
