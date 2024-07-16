module TestInterface

using BandedMatrices, LinearAlgebra, ArrayLayouts, FillArrays, Test, Random
import BandedMatrices: isbanded, AbstractBandedLayout, BandedStyle,
                        BandedColumns, bandeddata
import ArrayLayouts: OnesLayout, UnknownLayout
using InfiniteArrays

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
        @test bandwidths(Zeros(5,6)) == (-40320,-40320)
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

        @test isbanded(2Eye(5,6))
        @test bandwidths(2Eye(5,6)) == (0,0)
        @test BandedMatrices.inbands_getindex(2Eye(5,6), 1,1) == 2
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

        @test bandeddata(A) == bandeddata(Adjoint(A)) == bandeddata(Transpose(A)) == diag(A)'

        @test MemoryLayout(view(A, 1:3,2:4)) isa BandedColumns{DenseColumnMajor}
        @test MemoryLayout(view(A, [1,2,3],2:4)) isa UnknownLayout

        A[band(0)][1] = 3
        @test A[band(0)] == [2; ones(4)]

        B = Diagonal(Fill(1,5))
        @test @inferred(B[band(0)]) ≡ Fill(1,5)
        @test B[band(1)] ≡ B[band(-1)] ≡ Fill(0,4)
        @test B[band(2)] ≡ B[band(-2)] ≡ Fill(0,3)

        B = Diagonal(Ones(5))
        @test @inferred(B[band(0)]) ≡ Fill(1.0,5)
        @test B[band(1)] ≡ B[band(-1)] ≡ Fill(0.0,4)
        @test B[band(2)] ≡ B[band(-2)] ≡ Fill(0.0,3)

        B = Diagonal(1:∞)
        @test @inferred(B[Band(0)]) == 1:∞
    end

    @testset "SymTridiagonal" begin
        A = SymTridiagonal([1,2,3],[4,5])
        @test @inferred(A[Band(0)]) == [1,2,3]
        @test A[Band(1)] == A[Band(-1)] == [4,5]
        @test A[Band(2)] == A[Band(-2)] == [0]
        @test A[Band(3)] == A[Band(-3)] == Int[]
        @test isbanded(A)
        @test bandwidths(A) == (1,1)
        @test BandedMatrices.inbands_getindex(A, 1,1) == 1
        BandedMatrices.inbands_setindex!(A, 2, 1,1)
        @test A[1,1] == 2

        S = SymTridiagonal(1:∞, 1:∞)
        @test @inferred(S[Band(0)]) == S[Band(1)] == S[Band(-1)] == 1:∞

        B = SymTridiagonal(Fill(1,5), Fill(2,4))
        @test @inferred(B[band(0)]) ≡ Fill(1,5)
        @test B[band(1)] ≡ B[band(-1)] ≡ Fill(2,4)
        @test B[band(2)] ≡ B[band(-2)] ≡ Fill(0,3)

        B = SymTridiagonal(Ones(5), Ones(4))
        @test @inferred(B[band(0)]) ≡ Fill(1.0,5)
        @test B[band(1)] ≡ B[band(-1)] ≡ Fill(1.0,4)
        @test B[band(2)] ≡ B[band(-2)] ≡ Fill(0.0,3)
    end

    @testset "Tridiagonal" begin
        B = Tridiagonal([1:3;], [1:4;], [1:3;])
        @test @inferred(B[Band(0)]) == 1:4
        @test B[Band(1)] == B[Band(-1)] == 1:3
        @test B[Band(2)] == B[Band(-2)] == [0,0]
        @test B[Band(5)] == B[Band(-5)] == Int[]

        T = Tridiagonal(1:∞, 1:∞, 1:∞)
        @test @inferred(T[Band(0)]) == T[Band(1)] == T[Band(-1)] == 1:∞

        B = Tridiagonal(Fill(1,4), Fill(2,5), Fill(3,4))
        @test @inferred(B[band(0)]) ≡ Fill(2,5)
        @test B[band(1)] ≡ Fill(3,4)
        @test B[band(-1)] ≡ Fill(1,4)
        @test B[band(2)] ≡ B[band(-2)] ≡ Fill(0,3)
    end

    @testset "Bidiagonal" begin
        L = Bidiagonal([1:5;], [1:4;], :L)
        @test @inferred(L[Band(0)]) == 1:5
        @test L[Band(-1)] == 1:4
        @test L[Band(1)] == zeros(Int,4)

        L = Bidiagonal(1:∞, 1:∞, :L)
        @test @inferred(L[Band(0)]) == L[Band(-1)] == 1:∞

        L = Bidiagonal(Fill(2,5), Fill(1,4), :L)
        @test @inferred(L[band(0)]) ≡ Fill(2,5)
        @test L[band(1)] ≡ Fill(0,4)
        @test L[band(-1)] ≡ Fill(1,4)
        @test L[band(2)] ≡ L[band(-2)] ≡ Fill(0,3)
        @test BandedMatrix(L) == L

        U = Bidiagonal(Fill(2,5), Fill(1,4), :U)
        @test @inferred(U[band(0)]) ≡ Fill(2,5)
        @test U[band(1)] ≡ Fill(1,4)
        @test U[band(-1)] ≡ Fill(0,4)
        @test U[band(2)] ≡ U[band(-2)] ≡ Fill(0,3)
        @test BandedMatrix(U) == U
    end

    @testset "!Array" begin
        @test !BandedMatrices.isbanded(zeros(0,0))
    end

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

    @testset "copymutable_oftype" begin
        A = PseudoBandedMatrix(rand(5, 4), 1, 2)
        @test ArrayLayouts.copymutable_oftype_layout(MemoryLayout(A), A, BigFloat) == A
        @test ArrayLayouts.copymutable_oftype_layout(MemoryLayout(A), A, BigFloat) isa BandedMatrix
    end
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

@testset "OneElement" begin
    o = OneElement(1, 3, 5)
    @test bandwidths(o) == (2,-2)
    n,m = rand(1:10,2)
    o = OneElement(1, (rand(1:n),rand(1:m)), (n, m))
    @test bandwidths(o) == bandwidths(BandedMatrix(o))
    o = OneElement(1, (n+1,m+1), (n, m))
    @test bandwidths(o) == (-1, 0)
    o = OneElement(1, 6, 5)
    @test bandwidths(o) == (-1, 0)
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


@testset "permutedims" begin
    A = brand(10,11,1,2)
    @test bandwidths(permutedims(A)) == (2,1)
    @test permutedims(A) == permutedims(Matrix(A))

    @test permutedims(A') == permutedims(transpose(A)) == A

    B = A + im*A
    @test bandwidths(permutedims(B)) == (2,1)
    @test permutedims(B) == permutedims(Matrix(B))
    @test permutedims(B') == permutedims(Matrix(B)')
    @test permutedims(transpose(B)) == B

    A = BandedMatrix{Matrix{Float64}}(undef, (10, 11), (1, 2))
    A.data .= Ref([1 2; 3 4])
    # TODO: properly support PermutedDimsArray
    @test bandwidths(permutedims(A)) == (2,1)

    S = Symmetric(brand(10,10,1,2))
    @test permutedims(S) ≡ S
end

end # module
