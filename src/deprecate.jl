@deprecate gbmm!(α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T<:BlasFloat} gbmm!('N', 'N', α, A, B, β, C)


@deprecate BandedMatrix{T}(n::Integer,::Colon,a::Integer,b::Integer)  where {T} BandedMatrix{T}(uninitialized,n,:,a,b)
@deprecate BandedMatrix{T}(n::Integer,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix{T}(uninitialized, n, m, a, b)
@deprecate BandedMatrix{T}(n::Integer,a::Integer,b::Integer) where {T} BandedMatrix{T}(uninitialized, n, n, a, b)


@deprecate BandedMatrix(data::Matrix,m::Integer,a) BandedMatrices._BandedMatrix(data,m,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Integer,m::Integer,a) where {T} BandedMatrix{T}(uninitialized, n,m,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Integer,::Colon,a) where {T} BandedMatrix{T}(uninitialized, n,:,-a[1],a[end])
@deprecate BandedMatrix(::Type{T},n::Integer,a) where {T} BandedMatrix{T}(uninitialized, n,-a[1],a[end])

@deprecate BandedMatrix(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix{T}(uninitialized, n,m,a,b)


@deprecate BandedMatrix(::Type{T},n::Integer,a::Integer,b::Integer) where {T} BandedMatrix{T}(uninitialized, n,a,b)
@deprecate BandedMatrix(::Type{T},n::Integer,::Colon,a::Integer,b::Integer)  where {T} BandedMatrix{T}(uninitialized, n,:,a,b)

@deprecate BandedMatrix{T}(n::Integer,m::Integer,a) where {T} BandedMatrix{T}(uninitialized, n,m,-a[1],a[end])
@deprecate BandedMatrix{T}(n::Integer,::Colon,a) where {T} BandedMatrix{T}(uninitialized, n,:,-a[1],a[end])
@deprecate BandedMatrix{T}(n::Integer,a) where {T} BandedMatrix{T}(uninitialized, n,-a[1],a[end])

@deprecate bzeros(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Zeros{T}(n,m), (a,b))
@deprecate bzeros(::Type{T},n::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Zeros{T}(n,n), (a,b))
@deprecate bzeros(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) where {T} BandedMatrix(Zeros{T}(n,n+b), (a,b))
@deprecate bzeros(::Type{T},::Colon,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Zeros{T}(m+a,m), (a,b))
@deprecate bzeros(n::Integer,m::Integer,a::Integer,b::Integer) BandedMatrix(Zeros(n,m), (a,b))
@deprecate bzeros(n::Integer,a::Integer,b::Integer) BandedMatrix(Zeros(n,n), (a,b))

@deprecate bzeros(::Type{T},n::Integer,m::Integer,a) where {T}  BandedMatrix(Zeros(n,m),(-a[1],a[2]))
@deprecate bzeros(::Type{T},n::Number,::Colon,a) where {T} BandedMatrix(Zeros(n,n+a[2]),(-a[1],a[end]))
@deprecate bzeros(::Type{T},::Colon,m::Integer,a) where {T}  BandedMatrix(Zeros(m-a[1],m),(-a[1],a[end]))
@deprecate bzeros(::Type{T},n::Integer,a) where {T} BandedMatrix(Zeros{T}(n,n),(-a[1],a[end]))
@deprecate bzeros(n::Integer,m::Integer,a) BandedMatrix(Zeros(n,m),(-a[1],a[end]))
@deprecate bzeros(n::Integer,a) BandedMatrix(Zeros(n,n),(-a[1],a[end]))

@deprecate bzeros(B::AbstractMatrix) BandedMatrix(Zeros(B),bandwidths(B))


@deprecate bones(::Type{T},n::Integer,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Ones{T}(n,m), (a,b))
@deprecate bones(::Type{T},n::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Ones{T}(n,n), (a,b))
@deprecate bones(::Type{T},n::Integer,::Colon,a::Integer,b::Integer) where {T} BandedMatrix(Ones{T}(n,n+b), (a,b))
@deprecate bones(::Type{T},::Colon,m::Integer,a::Integer,b::Integer) where {T} BandedMatrix(Ones{T}(m+a,m), (a,b))
@deprecate bones(n::Integer,m::Integer,a::Integer,b::Integer) BandedMatrix(Ones(n,m), (a,b))
@deprecate bones(n::Integer,a::Integer,b::Integer) BandedMatrix(Ones(n,n), (a,b))

@deprecate bones(::Type{T},n::Integer,m::Integer,a) where {T}  BandedMatrix(Ones(n,m),(-a[1],a[2]))
@deprecate bones(::Type{T},n::Number,::Colon,a) where {T} BandedMatrix(Ones(n,n+a[2]),(-a[1],a[end]))
@deprecate bones(::Type{T},::Colon,m::Integer,a) where {T}  BandedMatrix(Ones(m-a[1],m),(-a[1],a[end]))
@deprecate bones(::Type{T},n::Integer,a) where {T} BandedMatrix(Ones{T}(n,n),(-a[1],a[end]))
@deprecate bones(n::Integer,m::Integer,a) BandedMatrix(Ones(n,m),(-a[1],a[end]))
@deprecate bones(n::Integer,a) BandedMatrix(Ones(n,n),(-a[1],a[end]))

@deprecate bones(B::AbstractMatrix) BandedMatrix(Ones(B),bandwidths(B))



@deprecate beye(::Type{T},n::Integer,a) where {T} BandedMatrix(Eye{T}(n), (-a[1],a[2]))
@deprecate beye(::Type{T},n::Integer) where {T} BandedMatrix(Eye{T}(n))
@deprecate beye(n::Integer,m::Integer,a::Integer,b::Integer) BandedMatrix(Eye(n,m),(a,b))
@deprecate beye(n::Integer) BandedMatrix(Eye(n))
@deprecate beye(n::Integer,a) BandedMatrix(Eye(n),(-a[1],a[2]))



@deprecate BandedMatrix{T}(data::Matrix{T},m::Integer,l::Int,u::Int) where T BandedMatrices._BandedMatrix(data,m,l,u)
@deprecate BandedMatrix(data::Matrix,m::Integer,l::Int,u::Int) BandedMatrices._BandedMatrix(data,m,l,u)

@deprecate SymBandedMatrix{T}(n::Integer,k::Integer) where {T} SymBandedMatrix{T}(uninitialized,n::Integer,k::Integer)

@deprecate SymBandedMatrix{T}(data::Matrix{T},k::Integer) where T BandedMatrices._SymBandedMatrix(data,k)
@deprecate SymBandedMatrix(data::Matrix,k::Integer) BandedMatrices._SymBandedMatrix(data,k)

@deprecate SymBandedMatrix(::Type{T},n::Integer,k::Integer) where {T} SymBandedMatrix{T}(uninitialized,n,k)


@deprecate sbzeros(::Type{T},n::Integer,a::Integer) where {T} SymBandedMatrix(Zeros{T}(n,n),a)
@deprecate sbzeros(n::Integer,a::Integer) SymBandedMatrix(Zeros(n,n),a)

@deprecate sbzeros(B::AbstractMatrix) SymBandedMatrix(Zeros(B), bandwidths(B))


@deprecate sbones(::Type{T},n::Integer,a::Integer) where {T} SymBandedMatrix(Ones{T}(n,n),a)
@deprecate sbones(n::Integer,a::Integer) SymBandedMatrix(Ones(n,n),a)

@deprecate sbones(B::AbstractMatrix) SymBandedMatrix(Ones(B), bandwidths(B))




@deprecate sbeye(::Type{T},n::Integer,a) where {T} SymBandedMatrix(Eye(n),a)
@deprecate sbeye(n::Integer) SymBandedMatrix(Eye(n))
