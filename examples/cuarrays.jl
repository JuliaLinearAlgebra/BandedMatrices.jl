####################
# This example demonstrates a BandedMatrix on the GPU using CLArrays
# We construct the Matrix on the CPU first for now as this is currently
# slow when tried to do directly on the GPU.
####################

using GPUArrays, CuArrays, FillArrays, BandedMatrices
import BandedMatrices: _BandedMatrix

function cu_finitedifference(n) where T
    data = CuArray{Float32}(3, n)
    data[2,:] .= -2f0 * n^2
    data[[1,3],:] = Δt/Δx^2
    _BandedMatrix(convert(T, data), n, 1, 1)
end


function expliciteuler(L, u₀, n)
    u = copy(u₀)
    v = copy(u)
    for _ = 1:n
        mul!(v, L, u)
        u,v = v,u
    end
    u
end

####
# 32 bit
####
nₓ = 20_000
Δx = 2f0/nₓ
Δt = Δx^2/8f0
T = 1f0 / 1_000
n_t = floor(Int, T/Δt)
L = finitedifference(CLArray, nₓ, Δt)

x= range(-1, stop=1, length=nₓ)
u₀ = CLArray{Float32}(exp.(-10x.^2));


@time begin
    u = expliciteuler(L, u₀, n_t);
    GPUArrays.synchronize(u)
end # 25s


L̃ = BandedMatrix{Float32,Matrix{Float32}}(L)
ũ₀ = Vector{Float32}(u₀)
@time ũ = expliciteuler(L̃, ũ₀, n_t); # 44s

maximum(abs, Array(u)-ũ)  # 3E-3


####
# 64 bit
####
nₓ = 20_000
Δx = 2/nₓ
Δt = Δx^2/8
T = 1f0 / 1_000
n_t = floor(Int, T/Δt)
L = finitedifference(CLArray, nₓ, Δt)

x= range(-1, stop=1, length=nₓ)
u₀ = CLArray(exp.(-10x.^2));


@time begin
    u = expliciteuler(L, u₀, n_t);
    GPUArrays.synchronize(u)
end # 30s

L̃ = BandedMatrix{Float64,Matrix{Float64}}(L)
ũ₀ = Vector{Float64}(u₀)
@time ũ = expliciteuler(L̃, ũ₀, n_t); # 45s

maximum(abs, Array(u)-ũ)  # 1.72E-13
