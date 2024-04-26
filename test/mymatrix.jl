# used to test general matrix backends
struct MyMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
end

MyMatrix{T}(::UndefInitializer, n::Int, m::Int) where T = MyMatrix{T}(Array{T}(undef, n, m))
MyMatrix(A::AbstractMatrix{T}) where T = MyMatrix{T}(Matrix{T}(A))
Base.convert(::Type{MyMatrix{T}}, A::MyMatrix{T}) where T = A
Base.convert(::Type{MyMatrix{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyMatrix}, A::MyMatrix)= A
Base.convert(::Type{AbstractArray{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{AbstractMatrix{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyMatrix{T}}, A::AbstractArray{T}) where T = MyMatrix{T}(A)
Base.convert(::Type{MyMatrix{T}}, A::AbstractArray) where T = MyMatrix{T}(convert(AbstractArray{T}, A))
Base.convert(::Type{MyMatrix}, A::AbstractArray{T}) where T = MyMatrix{T}(A)
Base.getindex(A::MyMatrix, kj...) = A.A[kj...]
Base.getindex(A::MyMatrix, ::Colon, j::AbstractVector) = MyMatrix(A.A[:,j])
Base.setindex!(A::MyMatrix, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyMatrix) = size(A.A)
Base.similar(::Type{MyMatrix{T}}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix, ::Type{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
