import BandedMatrices: banded_axpy!, banded_A_mul_B!

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

BandedMatrices.isbanded(::SimpleBandedMatrix) = true
BandedMatrices.bandwidth(A::SimpleBandedMatrix, k::Int) = k==1 ? A.l : A.u
BandedMatrices.inbands_getindex(A::SimpleBandedMatrix, j::Int, k::Int) = A.data[j, k]
BandedMatrices.inbands_setindex!(A::SimpleBandedMatrix, v, j::Int, k::Int) = setindex!(A.data, v, j, k)


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

@test banded_A_mul_B!(z, A, y) ≈ A*y ≈ Matrix(A)*y

B = SimpleBandedMatrix(rand(4, 4), 2, 3)
C = SimpleBandedMatrix(zeros(5, 4), 3, 4)
D = zeros(5, 4)

@test banded_A_mul_B!(C, A, B) ≈ banded_A_mul_B!(D, A, B) ≈ A*B
