using BandedMatrices, LinearAlgebra, LazyArrays, Test
import BandedMatrices: banded_axpy!, banded_mul!, isbanded, AbstractBandedLayout


struct SimpleBandedMatrix{T} <: AbstractMatrix{T}
    data::Array{T}
    l::Int
    u::Int
end


Base.size(A::SimpleBandedMatrix) = size(A.data)
function Base.getindex(A::SimpleBandedMatrix, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k]
    else
        zero(eltype(A.data))
    end
end
function Base.setindex!(A::SimpleBandedMatrix, v, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k] = v
    else
        error("out of band.")
    end
end

struct SimpleBandedLayout <: AbstractBandedLayout end
BandedMatrices.MemoryLayout(::SimpleBandedMatrix) = SimpleBandedLayout()
BandedMatrices.isbanded(::SimpleBandedMatrix) = true
BandedMatrices.bandwidth(A::SimpleBandedMatrix, k::Int) = k==1 ? A.l : A.u
BandedMatrices.inbands_getindex(A::SimpleBandedMatrix, j::Int, k::Int) = A.data[j, k]
BandedMatrices.inbands_setindex!(A::SimpleBandedMatrix, v, j::Int, k::Int) = setindex!(A.data, v, j, k)

@testset "banded matrix interface" begin
    @test isbanded(Zeros(5,6))
    @test bandwidths(Zeros(5,6)) == (0,0)
    @test BandedMatrices.inbands_getindex(Zeros(5,6), 1,2) == 0

    @test isbanded(Eye(5))
    @test bandwidths(Eye(5)) == (0,0)
    @test BandedMatrices.inbands_getindex(Eye(5), 1,1) == 1

    A = Diagonal(ones(5,5))
    @test isbanded(A)
    @test bandwidths(A) == (0,0)
    @test BandedMatrices.inbands_getindex(A, 1,1) == 1
    BandedMatrices.inbands_setindex!(A, 2, 1,1)
    @test A[1,1] == 2
    @test A[1,2] == 0
    @test BandedMatrices.@inbands(A[1,2]) == 2

    A = SymTridiagonal([1,2,3],[4,5])
    @test isbanded(A)
    @test bandwidths(A) == (1,1)
    @test BandedMatrices.inbands_getindex(A, 1,1) == 1
    BandedMatrices.inbands_setindex!(A, 2, 1,1)
    @test A[1,1] == 2

    A = SimpleBandedMatrix(rand(5, 4), 2, 2)
    B = rand(5, 4)
    C = copy(B)

    @test Matrix(banded_axpy!(2.0, A, B)) ≈ 2*Matrix(A) + C

    A = SimpleBandedMatrix(rand(5, 4), 1, 2)
    B = SimpleBandedMatrix(rand(5, 4), 2, 3)
    C = deepcopy(B)

    @test Matrix(banded_axpy!(2.0, A, C)) ≈ 2*Matrix(A) + B ≈ 2*A + Matrix(B) ≈ 2*Matrix(A) + Matrix(B) ≈ 2*A + B

    y = rand(4)
    z = zeros(5)
    z .= Mul(A, y)
    @test z ≈ A*y ≈ Matrix(A)*y

    B = SimpleBandedMatrix(rand(4, 4), 2, 3)
    C = SimpleBandedMatrix(zeros(5, 4), 3, 4)
    D = zeros(5, 4)

    @test (C .= Mul(A, B)) ≈ (D .= Mul(A, B)) ≈ A*B
end
