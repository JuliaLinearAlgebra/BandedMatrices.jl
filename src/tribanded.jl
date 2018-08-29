

isbanded(::AbstractTriangular{<:Any,<:AbstractBandedMatrix}) = true
bandwidths(A::UpperTriangular{<:Any,<:AbstractBandedMatrix}) = (min(0,bandwidth(parent(A),1)), bandwidth(parent(A),2))

triangularlayout(::Type{Tri}, ML::BandedColumnMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::BandedRowMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::ConjLayout{<:BandedRowMajor}) where {Tri} = Tri(ML)


Base.replace_in_print_matrix(A::AbstractTriangular{<:Any,<:AbstractBandedMatrix}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)



@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:InverseLayout{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor}},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'C', UNIT, triangulardata(A)', dest)
end
