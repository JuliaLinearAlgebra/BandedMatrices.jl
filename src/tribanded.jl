

isbanded(::AbstractTriangular{<:Any,<:AbstractBandedMatrix}) = true
bandwidths(A::Union{UpperTriangular{<:Any,<:AbstractBandedMatrix},UnitUpperTriangular{<:Any,<:AbstractBandedMatrix}}) =
    (min(0,bandwidth(parent(A),1)), bandwidth(parent(A),2))
bandwidths(A::Union{LowerTriangular{<:Any,<:AbstractBandedMatrix},UnitLowerTriangular{<:Any,<:AbstractBandedMatrix}}) =
    (bandwidth(parent(A),1), min(0,bandwidth(parent(A),2)))

triangularlayout(::Type{Tri}, ML::BandedColumnMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::BandedRowMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::ConjLayout{<:BandedRowMajor}) where {Tri} = Tri(ML)


function tribandeddata(::TriangularLayout{'U'}, A)
    B = triangulardata(A)
    u = bandwidth(B,2)
    D = bandeddata(B)
    view(D, 1:u+1, :)
end

function tribandeddata(::TriangularLayout{'L'}, A)
    B = triangulardata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    view(D, u+1:l+u+1, :)
end

tribandeddata(A) = tribandeddata(MemoryLayout(A), A)


Base.replace_in_print_matrix(A::Union{UpperTriangular{<:Any,<:AbstractBandedMatrix},
                                      UnitUpperTriangular{<:Any,<:AbstractBandedMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)

Base.replace_in_print_matrix(A::Union{LowerTriangular{<:Any,<:AbstractBandedMatrix},
                                      UnitLowerTriangular{<:Any,<:AbstractBandedMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)

# Mul
@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{'U',UNIT,<:BandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    tbmv!('U', 'N', UNIT, size(A,1), bandwidth(A,2), tribandeddata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{'L',UNIT,<:BandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    tbmv!('L', 'N', UNIT, size(A,1), bandwidth(A,1), tribandeddata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:BandedRowMajor},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    tbmv!(UPLO, 'T', UNIT, transpose(tribandeddata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:BandedRowMajor}},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    tbmv!(UPLO, 'C', UNIT, tribandeddata(A)', dest)
end


# Ldiv
@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:BandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    tbsv!('U', 'N', UNIT, size(A,1), bandwidth(A,2), tribandeddata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{'L',UNIT,<:BandedColumnMajor},
                                   <:AbstractStridedLayout}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    tbsv!('L', 'N', UNIT, size(A,1), bandwidth(A,1), tribandeddata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{UPLO,UNIT,<:BandedRowMajor},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    tbsv!(UPLO, 'T', UNIT, transpose(tribandeddata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         M::MatLdivVec{T, <:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:BandedRowMajor}},
                                   <:AbstractStridedLayout}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    tbsv!(UPLO, 'C', UNIT, tribandeddata(A)', dest)
end
