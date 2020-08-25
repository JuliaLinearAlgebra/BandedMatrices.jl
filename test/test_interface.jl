using BandedMatrices, LinearAlgebra, ArrayLayouts, FillArrays, Test, Base64
import BandedMatrices: banded_mul!, isbanded, AbstractBandedLayout, BandedStyle,
                        rowsupport, colsupport, _BandedMatrix, BandedColumns
import ArrayLayouts: OnesLayout, UnknownLayout

struct PseudoBandedMatrix{T} <: AbstractMatrix{T}
    data::Array{T}
    l::Int
    u::Int
end

Base.size(A::PseudoBandedMatrix) = size(A.data)
function Base.getindex(A::PseudoBandedMatrix, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k]
    else
        zero(eltype(A.data))
    end
end
function Base.setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k] = v
    else
        error("out of band.")
    end
end

struct PseudoBandedLayout <: AbstractBandedLayout end
Base.BroadcastStyle(::Type{<:PseudoBandedMatrix}) = BandedStyle()
BandedMatrices.MemoryLayout(::Type{<:PseudoBandedMatrix}) = PseudoBandedLayout()
BandedMatrices.isbanded(::PseudoBandedMatrix) = true
BandedMatrices.bandwidths(A::PseudoBandedMatrix) = (A.l , A.u)
BandedMatrices.inbands_getindex(A::PseudoBandedMatrix, j::Int, k::Int) = A.data[j, k]
BandedMatrices.inbands_setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int) = setindex!(A.data, v, j, k)
LinearAlgebra.fill!(A::PseudoBandedMatrix, v) = fill!(A.data,v)

@testset "banded matrix interface" begin
    @testset "Zeros" begin
        @test isbanded(Zeros(5,6))
        @test bandwidths(Zeros(5,6)) == (0,0)
        @test BandedMatrices.inbands_getindex(Zeros(5,6), 1,2) == 0
    end

    @testset "Eye" begin
        @test isbanded(Eye(5))
        @test bandwidths(Eye(5)) == (0,0)
        @test BandedMatrices.inbands_getindex(Eye(5), 1,1) == 1

        V = view(Eye(5),2:3,:)
        @test bandwidths(V) == (-1,1)
        @test MemoryLayout(typeof(V)) isa BandedColumns{OnesLayout}
        @test BandedMatrix(V) == [0 1 0 0 0; 0 0 1 0 0]

        B = brand(5,5,1,2)
        @test Eye(5) * B == B
        @test B * Eye(5) == B
        @test muladd!(2.0, Eye(5), B, 0.0, zeros(5,5)) == 2B
        @test muladd!(2.0, B, Eye(5), 0.0, zeros(5,5)) == 2B
    end

    @testset "Diagonal" begin
        A = Diagonal(ones(5,5))
        @test isbanded(A)
        @test bandwidths(A) == (0,0)
        @test BandedMatrices.inbands_getindex(A, 1,1) == 1
        BandedMatrices.inbands_setindex!(A, 2, 1,1)
        @test A[1,1] == 2
        @test A[1,2] == 0
        @test BandedMatrices.@inbands(A[1,2]) == 2

        @test MemoryLayout(view(A, 1:3,2:4)) isa BandedColumns{DenseColumnMajor}
        @test MemoryLayout(view(A, [1,2,3],2:4)) isa UnknownLayout
    end

    A = SymTridiagonal([1,2,3],[4,5])
    @test isbanded(A)
    @test bandwidths(A) == (1,1)
    @test BandedMatrices.inbands_getindex(A, 1,1) == 1
    BandedMatrices.inbands_setindex!(A, 2, 1,1)
    @test A[1,1] == 2

    A = PseudoBandedMatrix(rand(5, 4), 2, 2)
    B = rand(5, 4)
    C = copy(B)

    @test Matrix(B .= 2.0 .* A .+ B) ≈ 2*Matrix(A) + C

    A = PseudoBandedMatrix(rand(5, 4), 1, 2)
    B = (z -> exp(z)-1).(A)
    @test B isa BandedMatrix
    @test bandwidths(B) == bandwidths(A)
    @test B == (z -> exp(z)-1).(Matrix(A))

    A = PseudoBandedMatrix(rand(5, 4), 1, 2)
    B = A .* 2
    @test B isa BandedMatrix
    @test bandwidths(B) == bandwidths(A)
    @test B == 2Matrix(A) == (2 .* A)

    A = PseudoBandedMatrix(rand(5, 4), 1, 2)
    B = PseudoBandedMatrix(rand(5, 4), 2, 1)

    @test A .+ B isa BandedMatrix
    @test bandwidths(A .+ B) == (2,2)
    @test A .+ B == Matrix(A) + Matrix(B)

    A = PseudoBandedMatrix(rand(5, 4), 1, 2)
    B = PseudoBandedMatrix(rand(5, 4), 2, 3)
    C = deepcopy(B)
    @test Matrix(C .= 2.0 .* A .+ C) ≈ 2*Matrix(A) + B ≈ 2*A + Matrix(B) ≈ 2*Matrix(A) + Matrix(B) ≈ 2*A + B

    y = rand(4)
    z = zeros(5)
    muladd!(1.0, A, y, 0.0, z)
    @test z ≈ A*y ≈ Matrix(A)*y

    @test bandwidths(BandedMatrix(A)) ==
            bandwidths(BandedMatrix{Float64}(A)) ==
            bandwidths(BandedMatrix{Float64,Matrix{Float64}}(A)) ==
            bandwidths(convert(BandedMatrix{Float64}, A)) ==
            bandwidths(convert(BandedMatrix{Float64,Matrix{Float64}},A)) ==
            bandwidths(A)
end

@testset "Bi/Tri/Diagonal" begin
    D = Diagonal(randn(5))
    @test bandwidths(D) == (0,0)

    A =  brand(5,5,1,1)
    @test D*A isa BandedMatrix
    @test D*A == D*Matrix(A)

    @test A*D isa BandedMatrix
    @test A*D == Matrix(A)*D

    M = MulAdd(2.0, A, D, 1.0, A)
    @test copyto!(similar(M), M) ≈ 2A*D + A
    M = MulAdd(2.0, D, A, 1.0, A)
    @test copyto!(similar(M), M) ≈ 2D*A + A

    D = Diagonal(rand(Int,10))
    B = brand(10,10,-1,2)
    J = SymTridiagonal(randn(10), randn(9))
    T = Tridiagonal(randn(9), randn(10), randn(9))
    Bl = Bidiagonal(randn(10), randn(9), :L)
    Bu = Bidiagonal(randn(10), randn(9), :U)

    @test bandwidths(D) == (0,0)
    @test bandwidths(J) == bandwidths(T) == (1,1)
    @test bandwidths(Bu) == (0,1)
    @test bandwidths(Bl) == (1,0)

    @test isbanded(D)
    @test isbanded(J)
    @test isbanded(T)
    @test isbanded(Bu)
    @test isbanded(Bl)

    @test BandedMatrix(J) == J
    @test BandedMatrix(T) == T
    @test BandedMatrix(Bu) == Bu
    @test BandedMatrix(Bl) == Bl

    @test Base.BroadcastStyle(Base.BroadcastStyle(typeof(B)), Base.BroadcastStyle(typeof(D))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(D)), Base.BroadcastStyle(typeof(B))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(B)), Base.BroadcastStyle(typeof(T))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(T)), Base.BroadcastStyle(typeof(B))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(B)), Base.BroadcastStyle(typeof(J))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(J)), Base.BroadcastStyle(typeof(B))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(Bl)), Base.BroadcastStyle(typeof(B))) ==
        Base.BroadcastStyle(Base.BroadcastStyle(typeof(Bu)), Base.BroadcastStyle(typeof(B))) ==
            BandedStyle()

    A = B .+ D
    Ã = D .+ B
    @test A isa BandedMatrix
    @test Ã isa BandedMatrix
    @test bandwidths(A) == bandwidths(Ã) == (0,2)
    @test Ã == A == Matrix(B) + Matrix(D)

    A = B .+ J
    Ã = J .+ B
    @test A isa BandedMatrix
    @test Ã isa BandedMatrix
    @test bandwidths(A) == bandwidths(Ã) == (1,2)
    @test Ã == A == Matrix(B) + Matrix(J)

    A = B .+ T
    Ã = T .+ B
    @test A isa BandedMatrix
    @test Ã isa BandedMatrix
    @test bandwidths(A) == bandwidths(Ã) == (1,2)
    @test Ã == A == Matrix(B) + Matrix(T)

    A = B .+ Bl
    Ã = Bl .+ B
    @test A isa BandedMatrix
    @test Ã isa BandedMatrix
    @test bandwidths(A) == bandwidths(Ã) == (1,2)
    @test Ã == A == Matrix(B) + Matrix(Bl)

    A = B .+ Bu
    Ã = Bu .+ B
    @test A isa BandedMatrix
    @test Ã isa BandedMatrix
    @test bandwidths(A) == bandwidths(Ã) == (0,2)
    @test Ã == A == Matrix(B) + Matrix(Bu)

    @test layout_getindex(D,1:10,1:10) isa BandedMatrix
    @test layout_getindex(B,1:10,1:10) isa BandedMatrix
    @test layout_getindex(Bu,1:10,1:10) isa BandedMatrix
    @test layout_getindex(J,1:10,1:10) isa BandedMatrix
    @test layout_getindex(T,1:10,1:10) isa BandedMatrix
end

@testset "rot180" begin
    A = brand(5,5,1,2)
    R = rot180(A)
    @test bandwidths(R) == (2,1)
    @test R == rot180(Matrix(A))

    A = brand(5,4,1,2)
    R = rot180(A)
    @test bandwidths(R) == (3,0)
    @test R == rot180(Matrix(A))

    A = brand(5,6,1,-1)
    R = rot180(A)
    @test bandwidths(R) == (-2,2)
    @test R == rot180(Matrix(A))

    A = brand(6,5,-1,1)
    R = rot180(A)
    @test bandwidths(R) == (2,-2)
    @test R == rot180(Matrix(A))
end
