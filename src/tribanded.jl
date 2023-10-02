

isbanded(A::AbstractTriangular) = isbanded(parent(A))
bandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) =
    (min(0,bandwidth(parent(A),1)), bandwidth(parent(A),2))
bandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) =
    (bandwidth(parent(A),1), min(0,bandwidth(parent(A),2)))

triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedColumns} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedRows} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:ConjLayout{<:BandedRows}} = Tri{ML}()

sublayout(::TriangularLayout{UPLO,UNIT,ML}, inds) where {UPLO,UNIT,ML} = sublayout(ML(), inds)

getuplo(::Type{<:TriangularLayout{UPLO}}) where {UPLO} = UPLO
getunit(::Type{<:TriangularLayout{<:Any,UNIT}}) where {UNIT} = UNIT
getbwdim(uplo::Char) = uplo == 'U' ? 2 : 1

function bandeddata(::TriLayout, A) where {TriLayout<:TriangularLayout}
    B = triangulardata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    uplo = getuplo(TriLayout)
    rowinds = uplo == 'U' ? (1:u+1) : (u+1:l+u+1)
    view(D, rowinds, :)
end

# function bandedrowsdata(::TriangularLayout{'U'}, A)
#     B = triangulardata(A)
#     l,u = bandwidths(B)
#     D = bandedrowsdata(B)
#     view(D, :, l+1:l+u+1)
# end

# function bandeddata(::TriangularLayout{'L'}, A)
#     B = triangulardata(A)
#     l,_ = bandwidths(B)
#     D = bandedrowsdata(B)
#     view(D, :, 1:l+1)
# end

# Mul

copy(M::Lmul{<:TriangularLayout{uplo,trans,<:AbstractBandedLayout},<:AbstractBandedLayout}) where {uplo,trans} =
    ArrayLayouts.lmul!(M.A, BandedMatrix(M.B, bandwidths(M)))

@inline function materialize!(M::BlasMatLmulVec{<:TriLayout,
        <:AbstractStridedLayout}) where {UNIT, TriLayout<:TriangularLayout{<:Any,UNIT,<:BandedColumnMajor}}

    A,x = M.A,M.B
    uplo = getuplo(TriLayout)
    bwdim = getbwdim(uplo)
    tbmv!(uplo, 'N', UNIT, size(A,1), bandwidth(A,bwdim), bandeddata(A), x)
end

# @inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'U',UNIT,<:BandedRowMajor},
#                                                 <:AbstractStridedLayout}) where UNIT
#     A,x = M.A,M.B
#     tbmv!('U', 'T', UNIT, size(A,1), bandwidth(A,2), bandedrowsdata(A), x)
# end

# @inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'L',UNIT,<:BandedRowMajor},
#                                                 <:AbstractStridedLayout}) where UNIT
#     A,x = M.A,M.B
#     tbmv!('L', 'T', UNIT, size(A,1), bandwidth(A,1), bandeddata(transpose(parent(A))), x)
# end

# @inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:BandedRowMajor}},
#                                                 <:AbstractStridedLayout}) where {UPLO,UNIT}
#     A,x = M.A,M.B
#     tbmv!(UPLO, 'C', UNIT, bandeddata(A)', dest)
# end

# Ldiv
@inline function materialize!(M::BlasMatLdivVec{<:TriLayout,
        <:AbstractStridedLayout}) where {TriLayout<:TriangularLayout{<:Any,<:Any,<:BandedColumnMajor}}

    uplo = getuplo(TriLayout)
    unit = getunit(TriLayout)
    bwdim = getbwdim(uplo)
    A,x = M.A,M.B
    tbsv!(uplo, 'N', unit, size(A,1), bandwidth(A,bwdim), bandeddata(A), x)
end

@inline function materialize!(M::BlasMatLdivVec{<:TriLayout,
        <:AbstractStridedLayout}) where {TriLayout<:TriangularLayout{<:Any,<:Any,<:BandedRowMajor}}

    U,x = M.A,M.B
    A = triangulardata(U)
    uplo = getuplo(TriLayout)
    uploT = uplo == 'U' ? 'L' : 'U'
    unit = getunit(TriLayout)
    bwdim = getbwdim(uplo)
    LUT = uplo == 'U' ? LowerTriangular : UpperTriangular
    tbsv!(uploT, 'T', unit, size(A,1), bandwidth(A,bwdim), bandeddata(LUT(A')), x)
end


rot180(A::UpperTriangular{<:Any,<:AbstractBandedMatrix}) = LowerTriangular(rot180(parent(A)))
rot180(A::UnitUpperTriangular{<:Any,<:AbstractBandedMatrix}) = UnitLowerTriangular(rot180(parent(A)))
rot180(A::LowerTriangular{<:Any,<:AbstractBandedMatrix}) = UpperTriangular(rot180(parent(A)))
rot180(A::UnitLowerTriangular{<:Any,<:AbstractBandedMatrix}) = UnitUpperTriangular(rot180(parent(A)))