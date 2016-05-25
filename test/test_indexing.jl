using Base.Test
using BandedMatrices
import BandedMatrices: rowstart, rowstop, colstart, colstop, rowlength, collength

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

# scalar - integer - integer
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0

    # in band
    a[1, 1] = 0
    @test a[1, 1] == 0

    # out of band
    @test_throws BandError a[1, 3] = 0
    @test_throws BandError a[3, 1] = 0

    # out of range
    @test_throws BoundsError a[0, 0] = 0
    @test_throws BoundsError a[0, 1] = 0
    @test_throws BoundsError a[1, 0] = 0
    @test_throws BoundsError a[5, 6] = 0
    @test_throws BoundsError a[6, 5] = 0
    @test_throws BoundsError a[6, 6] = 0
end

# ~ indexing along a column

# scalar - colon - integer
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0

    # in band
    a[:, 1] = 2
    a[:, 2] = 3
    a[:, 3] = 4
    a[:, 4] = 5
    a[:, 5] = 6
    @test a ==  [2  3  0  0  0;
                 2  3  4  0  0;
                 0  3  4  5  0;
                 0  0  4  5  6;
                 0  0  0  5  6]

    @test_throws BoundsError a[:, 0] = 0
    @test_throws BoundsError a[:, 6] = 0
end

# vector - colon - integer
let
    a = bones(Int, 5, 7, 2, 1)
    # 5x7 BandedMatrices.BandedMatrix{Float64}:
    #  1.0  1.0    0    0    0    0   0   0
    #  1.0  1.0  1.0    0    0    0   0   0
    #  1.0  1.0  1.0  1.0    0    0   0   0
    #    0  1.0  1.0  1.0  1.0    0   0   0
    #    0    0  1.0  1.0  1.0  1.0   0   0

    # in band
    a[:, 1] = [1,   2,  3]
    a[:, 2] = [4,   5,  6,  7]
    a[:, 3] = [8,   9, 10, 11]
    a[:, 4] = [12, 13, 14]
    a[:, 5] = [15, 16]
    a[:, 6] = [17]

    @test a == [ 1  4   0   0   0   0   0;
                 2  5   8   0   0   0   0;
                 3  6   9  12   0   0   0;
                 0  7  10  13  15   0   0;
                 0  0  11  14  16  17   0]

    @test_throws BoundsError a[:, 0] = [1, 2, 3]
    @test_throws BoundsError a[:, 8] = [1, 2, 3]
    @test_throws DimensionMismatch a[:, 1] = [1, 2]
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
end


# ~ indexing along a row

# scalar - integer - colon
let
    a = bones(5, 5, 1, 1)
    # 1.0  1.0  0.0  0.0  0.0
    # 1.0  1.0  1.0  0.0  0.0
    # 0.0  1.0  1.0  1.0  0.0
    # 0.0  0.0  1.0  1.0  1.0
    # 0.0  0.0  0.0  1.0  1.0    

    # in band
    a[1, :] = 2
    a[2, :] = 3
    a[3, :] = 4
    a[4, :] = 5
    a[5, :] = 6
    @test a == [2  2  0  0  0;
                3  3  3  0  0;
                0  4  4  4  0;
                0  0  5  5  5;
                0  0  0  6  6]


    @test_throws BoundsError a[0, :] = 0
    @test_throws BoundsError a[6, :] = 0
end

# vector - integer - colon
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
    a[1, :] = [1,   2,  3]
    a[2, :] = [4,   5,  6,  7]
    a[3, :] = [8,   9, 10, 11]
    a[4, :] = [12, 13, 14]
    a[5, :] = [15, 16]
    a[6, :] = [17]

    @test a ==  [1  2   3   0   0;
                 4  5   6   7   0;
                 0  8   9  10  11;
                 0  0  12  13  14;
                 0  0   0  15  16;
                 0  0   0   0  17;
                 0  0   0   0   0]

    @test_throws BoundsError a[0, :] = [1, 2, 3]
    @test_throws BoundsError a[8, :] = [1, 2, 3]
    @test_throws DimensionMismatch a[1, :] = [1, 2]
    @test_throws BoundsError a[7, :] = [1, 2, 3]
    @test_throws BoundsError a[7, :] = 1
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
    a[1, :] = 2
    a[2, :] = 3
    a[3, :] = 4
    a[4, :] = 5
    a[5, :] = 6
    a[6, :] = 7

    @test a ==  [2  2  2  0  0;
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

# other methods 
let
    a = bzeros(3, 3, 1, 1)
    a[:, :] = 1
    @test a == [1 1 0;
                1 1 1;
                0 1 1]
end