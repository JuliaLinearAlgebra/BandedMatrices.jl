####################
# This example demonstrates a BandedMatrix on the GPU using CLArrays
# We construct the Matrix on the CPU first for now as this is currently
# slow when tried to do directly on the GPU.
####################

using GPUArrays, CLArrays, FillArrays, BandedMatrices, Plots
import BandedMatrices: _BandedMatrix

function finitedifference(::Type{T}, n, Δt) where T
    Δx = 2/n
    data = Array{Float32}(3, n)
    data[2,:] = 1-2*Δt/Δx^2
    data[[1,3],:] = Δt/Δx^2
    _BandedMatrix(convert(T, data), n, 1, 1)
end


function expliciteuler(L, u₀, n)
    u = copy(u₀)
    v = copy(u)
    for _ = 1:n
        A_mul_B!(v, L, u)
        u = v
    end
    u
end


# For some unknown reason, the code above has a bug and breaks with
# standard banded matrix
function expliciteuler2(L, u₀, n)
    u = copy(u₀)
    for _ = 1:n
        u = L*u
    end
    u
end



nₓ = 20_000
Δx = 2/nₓ
Δt = Δx^2/8
T = 1f0 / 1_000
n_t = floor(Int, T/Δt)
L = finitedifference(CLArray, nₓ, Δt)

x= linspace(-1, 1, nₓ)
u₀ = CLArray{Float32}(exp.(-10x.^2));


@time u = expliciteuler(L, u₀, n_t);  # 16s

L̃ = BandedMatrix{Float32,Matrix{Float32}}(L)
ũ₀ = Vector{Float32}(u₀)
@time ũ = expliciteuler2(L̃, ũ₀, n_t); # 51s

maximum(abs, Array(u)-ũ)  # 3E-3

plot(x, Array(u₀))
    plot!(x, Array(u))
    plot!(x, ũ)
