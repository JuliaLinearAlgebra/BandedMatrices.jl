

isbanded(A::AbstractTriangular) = isbanded(parent(A))
bandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) =
    (min(0,bandwidth(parent(A),1)), bandwidth(parent(A),2))
bandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) =
    (bandwidth(parent(A),1), min(0,bandwidth(parent(A),2)))

triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedColumns} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedRows} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:ConjLayout{<:BandedRows}} = Tri{ML}()

sublayout(::TriangularLayout{UPLO,UNIT,ML}, inds) where {UPLO,UNIT,ML} = sublayout(ML(), inds)


function bandeddata(::TriangularLayout{'U'}, A)
    B = triangulardata(A)
    u = bandwidth(B,2)
    D = bandeddata(B)
    view(D, 1:u+1, :)
end

function bandeddata(::TriangularLayout{'L'}, A)
    B = triangulardata(A)
    l,u = bandwidths(B)
    D = bandeddata(B)
    view(D, u+1:l+u+1, :)
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

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'U',UNIT,<:BandedColumnMajor},
                                                <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    tbmv!('U', 'N', UNIT, size(A,1), bandwidth(A,2), bandeddata(A), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'L',UNIT,<:BandedColumnMajor},
                                                <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    tbmv!('L', 'N', UNIT, size(A,1), bandwidth(A,1), bandeddata(A), x)
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
for UNIT in ('N', 'U')
    @eval begin
        @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',$UNIT,<:BandedColumnMajor},
                                        <:AbstractStridedLayout})
            A,x = M.A,M.B
            tbsv!('U', 'N', $UNIT, size(A,1), bandwidth(A,2), bandeddata(A), x)
        end

        @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',$UNIT,<:BandedColumnMajor},
                                                        <:AbstractStridedLayout})
            A,x = M.A,M.B
            tbsv!('L', 'N', $UNIT, size(A,1), bandwidth(A,1), bandeddata(A), x)
        end
    end
    # for UPLO in ('U', 'L')
    #     @eval begin
    #         @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{$UPLO,$UNIT,BandedRowMajor},
    #                                                     <:AbstractStridedLayout})
    #             A,x = M.A,M.B
    #             tbsv!($UPLO, 'T', $UNIT, transpose(bandeddata(A)), x)
    #         end

    #         @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{$UPLO,$UNIT,ConjLayout{BandedRowMajor}},
    #                                                     <:AbstractStridedLayout})
    #             A,x = M.A,M.B
    #             tbsv!($UPLO, 'C', $UNIT, bandeddata(A)', x)
    #         end
    #     end
    # end
end

