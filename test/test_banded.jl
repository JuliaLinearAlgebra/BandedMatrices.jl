import BandedMatrices: _BandedMatrix

# some basic operations

@test BandedMatrix(Zeros(5,5), (1,1)) == _BandedMatrix(zeros(3,5), 5, 1, 1)
@test BandedMatrix(Zeros{Int}(5,5), (1,1)) == _BandedMatrix(zeros(Int,3,5), 5, 1, 1)
@test BandedMatrix{Int}(Zeros(5,5), (1,1)) == _BandedMatrix(zeros(Int,3,5), 5, 1, 1)

@test_throws UndefRefError BandedMatrix{Vector{Float64}}(uninitialized, (5,5), (1,1))[1,1]


let A = brand(10,12,2,3),B = brand(10,12,3,4)
    @test Matrix(sparse(A)) ≈ Matrix(A)

    @test Matrix(A') ≈ Matrix(A)'
    @test Matrix(A.') ≈ Matrix(A).'
    @test Matrix((A+im*A)') ≈ (Matrix(A)+im*Matrix(A))'
    @test Matrix((A+im*A).') ≈ (Matrix(A)+im*Matrix(A)).'

    @test Matrix(A+B) ≈ (Matrix(A)+Matrix(B))
    @test Matrix(A-B) ≈ (Matrix(A)-Matrix(B))

    @test Matrix(A.*B) ≈ (Matrix(A).*Matrix(B))
end


# banded * vec

let A=brand(10,12,2,3), v=rand(12), w=rand(10)
    @test A*v ≈ Matrix(A)*v
    @test A'*w ≈ Matrix(A)'*w
end

let A=brand(Float64,5,3,2,2), v=rand(Complex128,3), w=rand(Complex128,5)
    @test A*v ≈ Matrix(A)*v
    @test A'*w ≈ Matrix(A)'*w
end

let A=brand(Complex128,5,3,2,2), v=rand(Complex128,3), w=rand(Complex128,5)
    @test A*v ≈ Matrix(A)*v
    @test A'*w ≈ Matrix(A)'*w
end

let A=brand(Complex128,5,3,2,2), v=rand(Float64,3), w=rand(Float64,5)
    @test A*v ≈ Matrix(A)*v
    @test A'*w ≈ Matrix(A)'*w
end


# test matrix multiplications

# big banded * dense

let A=brand(1000,1000,200,300), B=rand(1000,1000)
    @test A*B ≈ Matrix(A)*B
    @test B*A ≈ B*Matrix(A)
end
# gbmm! not yet implemented
# @test A'*B ≈ Matrix(A)'*B
# @test A*B' ≈ Matrix(A)*B'
# @test A'*B' ≈ Matrix(A)'*B'

let A=brand(1200,1000,200,300), B=rand(1000,1000), C=rand(1200,1200)
    @test A*B ≈ Matrix(A)*B
    @test C*A ≈ C*Matrix(A)
end
# gbmm! not yet implemented
# @test A'*C ≈ Matrix(A)'*C
# @test A*B' ≈ Matrix(A)*B'
# @test A'*C' ≈ Matrix(A)'*C'


# banded * banded


for n in (1,5), ν in (1,5), m in (1,5), Al in (0,1,3), Au in (0,1,3),
        Bl in (0,1,3), Bu in (0,1,3)
    let A = brand(n, ν, Al, Au), B = brand(ν, m, Bl, Bu),
            C = brand(ν, n, Al, Bu), D = brand(m, ν, Al, Bu)
        @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
        @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
        @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
        @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
    end
end

let A = brand(Complex128, 5, 4, 2, 3), B = brand(Complex128, 4, 6, 3, 1),
    C = brand(Complex128, 4, 5, 1, 1), D = brand(Complex128, 6, 4, 0, 3)
    @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
    @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
    @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
    @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
end

let A = brand(Complex128, 5, 4, 2, 3), B = brand(4, 6, 3, 1), C = brand(4, 5, 1, 1),
        D = brand(Complex128, 6, 4, 0, 3)
    @test Matrix(A*B) ≈ Matrix(A)*Matrix(B)
    @test Matrix(C'*B) ≈ Matrix(C)'*Matrix(B)
    @test Matrix(A*D') ≈ Matrix(A)*Matrix(D)'
    @test Matrix(C'*D') ≈ Matrix(C)'*Matrix(D)'
end


## BigFloat

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


## negative bands

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




## UniformScaling
let A = brand(10,10,1,2)
    @test Matrix(A+I) ≈ Matrix(A)+I
    @test Matrix(I-A) ≈ I-Matrix(A)


    # number
    @test Matrix(A/2) ≈ Matrix(A)/2
end

# zero arrays

let b = rand(4)
    for A in (brand(3,4,-1,0),brand(3,4,0,-1),brand(3,4,-1,-1)),
        B in (brand(4,3,1,2),brand(4,3,-1,0),brand(4,3,-1,-1))
        @test Matrix(A) == zeros(3,4)
        @test A*B == zeros(3,3)
        @test A*b == zeros(3)
    end
end


# check that col/rowstop is ≥ 0
let A = brand(3,4,-2,2)
    @test BandedMatrices.colstop(A, 1) == BandedMatrices.colstop(A, 2) == 0
    @test BandedMatrices.colstop(A, 3) == 1
end

let A = brand(3,4,2,-2)
    @test BandedMatrices.rowstop(A, 1) == BandedMatrices.rowstop(A, 2) == 0
    @test BandedMatrices.rowstop(A, 3) == 1
end

 # Test for errors in collect

let B=brand(10,10,0,4)
    @test B*[collect(1.0:10) collect(1.0:10)] ≈ Matrix(B)*[collect(1.0:10) collect(1.0:10)]
end
