using Base.Test
using BandedMatrices

# collength/rowlength
let
    a = bzeros(5, 5, 1, 1)
    #  0.0  0.0
    #  0.0  0.0  0.0
    #       0.0  0.0  0.0
    #            0.0  0.0  0.0
    #                 0.0  0.0
    @test_throws BoundsError BandedMatrices.collength(a, 0)
    @test_throws BoundsError BandedMatrices.rowlength(a, 0)
    @test                    BandedMatrices.collength(a, 1) == 2
    @test                    BandedMatrices.collength(a, 2) == 3
    @test                    BandedMatrices.collength(a, 3) == 3
    @test                    BandedMatrices.collength(a, 4) == 3
    @test                    BandedMatrices.collength(a, 5) == 2
    @test                    BandedMatrices.rowlength(a, 1) == 2
    @test                    BandedMatrices.rowlength(a, 2) == 3
    @test                    BandedMatrices.rowlength(a, 3) == 3
    @test                    BandedMatrices.rowlength(a, 4) == 3
    @test                    BandedMatrices.rowlength(a, 5) == 2
    @test_throws BoundsError BandedMatrices.collength(a, 6)
    @test_throws BoundsError BandedMatrices.rowlength(a, 6)

    a = bzeros(5, 5, 1, 2)
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0  0.0
    #      0.0  0.0  0.0  0.0
    #           0.0  0.0  0.0
    #                0.0  0.0
    @test_throws BoundsError BandedMatrices.collength(a, 0)
    @test_throws BoundsError BandedMatrices.rowlength(a, 0)
    @test                    BandedMatrices.collength(a, 1) == 2
    @test                    BandedMatrices.collength(a, 2) == 3
    @test                    BandedMatrices.collength(a, 3) == 4
    @test                    BandedMatrices.collength(a, 4) == 4
    @test                    BandedMatrices.collength(a, 5) == 3
    @test                    BandedMatrices.rowlength(a, 1) == 3
    @test                    BandedMatrices.rowlength(a, 2) == 4
    @test                    BandedMatrices.rowlength(a, 3) == 4
    @test                    BandedMatrices.rowlength(a, 4) == 3
    @test                    BandedMatrices.rowlength(a, 5) == 2
    @test_throws BoundsError BandedMatrices.collength(a, 6)
    @test_throws BoundsError BandedMatrices.rowlength(a, 6)

    a = bzeros(5, 7, 2, 1)
    # 0.0  0.0
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0  0.0
    #      0.0  0.0  0.0  0.0
    #           0.0  0.0  0.0  0.0
    @test_throws BoundsError BandedMatrices.collength(a, 0)
    @test_throws BoundsError BandedMatrices.rowlength(a, 0)
    @test                    BandedMatrices.collength(a, 1) == 3
    @test                    BandedMatrices.collength(a, 2) == 4
    @test                    BandedMatrices.collength(a, 3) == 4
    @test                    BandedMatrices.collength(a, 4) == 3
    @test                    BandedMatrices.collength(a, 5) == 2
    @test                    BandedMatrices.collength(a, 6) == 1
    @test                    BandedMatrices.collength(a, 7) == 0
    @test                    BandedMatrices.rowlength(a, 1) == 2
    @test                    BandedMatrices.rowlength(a, 2) == 3
    @test                    BandedMatrices.rowlength(a, 3) == 4
    @test                    BandedMatrices.rowlength(a, 4) == 4
    @test                    BandedMatrices.rowlength(a, 5) == 4
    @test_throws BoundsError BandedMatrices.collength(a, 8)
    @test_throws BoundsError BandedMatrices.rowlength(a, 6)
end

# scalar - integer - integer
let
    a = bones(5, 5, 1, 1)

    # in band
    a[1, 1] = 0
    @test a[1, 1] == 0

    # out of band
    @test_throws ErrorException a[1, 3] = 0
    @test_throws ErrorException a[3, 1] = 0

    # out of range
    @test_throws BoundsError a[0, 0] = 0
    @test_throws BoundsError a[0, 1] = 0
    @test_throws BoundsError a[1, 0] = 0
    @test_throws BoundsError a[5, 6] = 0
    @test_throws BoundsError a[6, 5] = 0
    @test_throws BoundsError a[6, 6] = 0
end


# scalar - colon - integer
let
    a = bones(5, 5, 1, 1)

    # in band
    a[:, 1] = 2
    @test a[1, 1] == 2
    @test a[2, 1] == 2
    @test a[3, 1] == 0
    @test a[4, 1] == 0
    @test a[5, 1] == 0

    a[:, 3] = 4
    @test a[1, 3] == 0
    @test a[2, 3] == 4
    @test a[3, 3] == 4
    @test a[4, 3] == 4
    @test a[5, 3] == 0

    a[:, 5] = 5
    @test a[1, 5] == 0
    @test a[2, 5] == 0
    @test a[3, 5] == 0
    @test a[4, 5] == 5
    @test a[5, 5] == 5

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
    a = bzeros(3, 4, 2, 1)

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
    @test_throws BoundsError a[0:1, 1] = 3
    @test_throws BoundsError a[1:4, 1] = 3
    @test_throws BoundsError a[2:3, 4] = 3
    @test_throws BoundsError a[3:4, 4] = 3

end

# vector - range - integer
let
    a = bzeros(5, 4, 1, 2)
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