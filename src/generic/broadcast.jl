
struct BandedStyle <: AbstractArrayStyle{2} end
BandedStyle(::Val{2}) = BandedStyle()
BroadcastStyle(::Type{<:AbstractBandedMatrix}) = BandedStyle()
BroadcastStyle(::DefaultArrayStyle{0}, ::BandedStyle) = BandedStyle()

copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(identity)}) =
    banded_copyto!(dest, bc.args...)

copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, typeof(identity)}) =
    banded_copyto!(dest, bc.args...)


A = brand(n,n,1,1)
    B = brand(n,n,2,2)
    @time B .= 2.0 .* A

BroadcastStyle(A)

y .= a .* x .+ y

# copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle}) =
#     copyto!(dest, Broadcasted{DefaultArrayStyle{2}()}(bc.f, bc.args, bc.axes))

# copy(bc::Broadcasted{BandedStyle}) =
#     copy(Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes))
