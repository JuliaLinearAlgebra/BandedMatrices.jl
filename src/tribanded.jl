

isbanded(A::AbstractTriangular) = isbanded(parent(A))
bandwidths(A::Union{UpperTriangular,UnitUpperTriangular}) =
    (min(0,bandwidth(parent(A),1)), bandwidth(parent(A),2))
bandwidths(A::Union{LowerTriangular,UnitLowerTriangular}) =
    (bandwidth(parent(A),1), min(0,bandwidth(parent(A),2)))

triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedColumns} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:BandedRows} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri,ML<:ConjLayout{<:BandedRows}} = Tri{ML}()


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

tribandeddata(A) = tribandeddata(MemoryLayout(typeof(A)), A)


Base.replace_in_print_matrix(A::Union{UpperTriangular{<:Any,<:AbstractBandedMatrix},
                                      UnitUpperTriangular{<:Any,<:AbstractBandedMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)

Base.replace_in_print_matrix(A::Union{LowerTriangular{<:Any,<:AbstractBandedMatrix},
                                      UnitLowerTriangular{<:Any,<:AbstractBandedMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    -bandwidth(A,1) ≤ j-i ≤ bandwidth(A,2) ? s : Base.replace_with_centered_mark(s)

# Mul
@lazylmul UpperTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazylmul UnitUpperTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazylmul LowerTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazylmul UnitLowerTriangular{T, <:AbstractBandedMatrix{T}} where T


@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'U',UNIT,<:BandedColumnMajor},
                                                <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    tbmv!('U', 'N', UNIT, size(A,1), bandwidth(A,2), tribandeddata(A), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'L',UNIT,<:BandedColumnMajor},
                                                <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    tbmv!('L', 'N', UNIT, size(A,1), bandwidth(A,1), tribandeddata(A), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,BandedRowMajor},
                                                <:AbstractStridedLayout}) where {UPLO,UNIT}
    A,x = M.A,M.B
    tbmv!(UPLO, 'T', UNIT, transpose(tribandeddata(A)), x)
end


@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,ConjLayout{BandedRowMajor}},
                                                <:AbstractStridedLayout}) where {UPLO,UNIT}
    A,x = M.A,M.B
    tbmv!(UPLO, 'C', UNIT, tribandeddata(A)', dest)
end

# Ldiv
@lazyldiv UpperTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazyldiv UnitUpperTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazyldiv LowerTriangular{T, <:AbstractBandedMatrix{T}} where T
@lazyldiv UnitLowerTriangular{T, <:AbstractBandedMatrix{T}} where T

for UNIT in ('N', 'U')
    @eval begin
        @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',$UNIT,<:BandedColumnMajor},
                                        <:AbstractStridedLayout})
            A,x = M.A,M.B
            tbsv!('U', 'N', $UNIT, size(A,1), bandwidth(A,2), tribandeddata(A), x)
        end

        @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',$UNIT,<:BandedColumnMajor},
                                                        <:AbstractStridedLayout})
            A,x = M.A,M.B
            tbsv!('L', 'N', $UNIT, size(A,1), bandwidth(A,1), tribandeddata(A), x)
        end
    end
    for UPLO in ('U', 'L')
        @eval begin
            @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{$UPLO,$UNIT,BandedRowMajor},
                                                        <:AbstractStridedLayout})
                A,x = M.A,M.B
                tbsv!($UPLO, 'T', $UNIT, transpose(tribandeddata(A)), x)
            end

            @inline function materialize!(M::BlasMatLdivVec{<:TriangularLayout{$UPLO,$UNIT,ConjLayout{BandedRowMajor}},
                                                        <:AbstractStridedLayout})
                A,x = M.A,M.B
                tbsv!($UPLO, 'C', $UNIT, tribandeddata(A)', x)
            end
        end
    end
end

