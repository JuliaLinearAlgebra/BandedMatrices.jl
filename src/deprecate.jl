@deprecate gbmm!(α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T<:BlasFloat} gbmm!('N', 'N', α, A, B, β, C)


@deprecate BandedMatrix(data::Matrix,m::Int,a) BandedMatrix(data,m,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Int,m::Int,a) where {T} BandedMatrix(T,n,m,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Int,::Colon,a) where {T} BandedMatrix(T,n,:,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Int,a) where {T} BandedMatrix(T,n,-a[1],a[end])




@deprecate bzeros(::Type{T},n::Int,m::Int,a::Int,b::Int) where {T} BandedMatrix(Zeros{T}(n,m), (a,b))
@deprecate bzeros(::Type{T},n::Int,a::Int,b::Int) where {T} BandedMatrix(Zeros{T}(n,n), (a,b))
@deprecate bzeros(::Type{T},n::Int,::Colon,a::Int,b::Int) where {T} BandedMatrix(Zeros{T}(n,n+b), (a,b))
@deprecate bzeros(::Type{T},::Colon,m::Int,a::Int,b::Int) where {T} BandedMatrix(Zeros{T}(m+a,m), (a,b))
@deprecate bzeros(n::Int,m::Int,a::Int,b::Int) BandedMatrix(Zeros(n,m), (a,b))
@deprecate bzeros(n::Int,a::Int,b::Int) BandedMatrix(Zeros(n,n), (a,b))

@deprecate bzeros(::Type{T},n::Int,m::Int,a) where {T}  BandedMatrix(Zeros(n,m),(-a[1],a[2]))
@deprecate bzeros(::Type{T},n::Number,::Colon,a) where {T} BandedMatrix(Zeros(n,n+a[2]),(-a[1],a[end]))
@deprecate bzeros(::Type{T},::Colon,m::Int,a) where {T}  BandedMatrix(Zeros(m-a[1],m),(-a[1],a[end]))
@deprecate bzeros(::Type{T},n::Int,a) where {T} BandedMatrix(Zeros{T}(n,n),(-a[1],a[end]))
@deprecate bzeros(n::Int,m::Int,a) BandedMatrix(Zeros(n,m),(-a[1],a[end]))
@deprecate bzeros(n::Int,a) BandedMatrix(Zeros(n,n),(-a[1],a[end]))

@deprecate bzeros(B::AbstractMatrix) BandedMatrix(Zeros(B),bandwidths(B))



@deprecate beye(::Type{T},n::Int,a) where {T} BandedMatrix(Eye{T}(n), (-a[1],a[2]))
@deprecate beye(::Type{T},n::Int) where {T} BandedMatrix(Eye{T}(n))
@deprecate beye(n::Int) BandedMatrix(Eye(n))
@deprecate beye(n::Int,a) BandedMatrix(Eye(n),(-a[1],a[2]))

@deprecate BandedMatrix(data::Matrix,n::Int,l::Int,u::Int) BandedMatrix{eltype(data)}(data,n,l,u)
@deprecate SymBandedMatrix(data::Matrix,k::Int) SymBandedMatrix{eltype(data)}(data,k)

@deprecate sbzeros(::Type{T},n::Int,a::Int) where {T} SymBandedMatrix(Zeros{T}(n,n),a)
@deprecate sbzeros(n::Int,a::Int) SymBandedMatrix(Zeros(n,n),a)

# @deprecate sbzeros(B::AbstractMatrix) SymBandedMatrix(Zeros(B), bandwidths(B))
