struct BandedEigenvectors{T} <: AbstractMatrix{T}
    Q::Matrix{T}
    G::Vector{Givens{T}}
end

size(B::BandedEigenvectors) = size(B.Q)

getindex(B::BandedEigenvectors, i, j) = Matrix(B)[i, j]

function bandedeigen(A::Symmetric{T,M}) where {T,M<:BandedMatrix{T}}
    N = size(A, 1)
    KD = bandwidth(A)
    D = Vector{T}(undef, N)
    E = Vector{T}(undef, N-1)
    G = Vector{Givens{T}}(undef, 0)
    WORK = Vector{T}(undef, N)
    AB = Matrix(symbandeddata(A))
    sbtrd!('V', A.uplo, N, KD, AB, D, E, G, WORK)
    F = eigen(SymTridiagonal(D, E))
    Eigen(F.values, BandedEigenvectors(F.vectors, G))
end

function Matrix(B::BandedEigenvectors)
    Q = copy(B.Q)
    G = B.G
    for k in length(G):-1:1
        lmul!(G[k]', Q)
    end
    return Q
end

function mul!(y::Array{T,N}, B::BandedEigenvectors{T}, x::Array{T,N}) where {T,N}
    mul!(y, B.Q, x)
    G = B.G
    for k in length(G):-1:1
        lmul!(G[k]', y)
    end
    y
end

function mul!(y::Array{T,N}, B::Adjoint{T,BandedEigenvectors{T}}, x::Array{T,N}) where {T,N}
    z = copy(x)
    G = B.parent.G
    for k in 1:length(G)
        lmul!(G[k], z)
    end
    mul!(y, B.parent.Q', z)
    y
end

function mul!(y::Array{T,N}, B::Transpose{T,BandedEigenvectors{T}}, x::Array{T,N}) where {T,N}
    z = copy(x)
    G = B.parent.G
    for k in 1:length(G)
        lmul!(G[k], z)
    end
    mul!(y, transpose(B.parent.Q), z)
    y
end


#  Definition:
#  ===========
#
#       SUBROUTINE DSBTRD( VECT, UPLO, N, KD, AB, LDAB, D, E, Q, LDQ,
#                          WORK, INFO )
#
#       .. Scalar Arguments ..
#       CHARACTER          UPLO, VECT
#       INTEGER            INFO, KD, LDAB, LDQ, N
#       ..
#       .. Array Arguments ..
#       DOUBLE PRECISION   AB( LDAB, * ), D( * ), E( * ), Q( LDQ, * ),
#      $                   WORK( * )
#       ..
function sbtrd!(VECT::Char, UPLO::Char,
                N::Int, KD::Int, AB::AbstractMatrix{T},
                D::AbstractVector{T}, E::AbstractVector{T}, Q::Vector{Givens{T}},
                WORK::AbstractVector{T}) where T
    require_one_based_indexing(AB)
    chkstride1(AB)
    chkuplo(UPLO)
    chkvect(VECT)
    M    = size(AB,2)
    M ≠ N && throw(ArgumentError("Matrix must be square"))
    size(AB,1) < KD+1 && throw(ArgumentError("Not enough bands"))

    LDAB = max(1, stride(AB, 2))

    ZERO = zero(T)
    TEMP = Ref{T}()
    refD = Ref{T}()
    refWORK = Ref{T}()
    KD1 = KD + 1
    KDM1 = KD - 1
    INCX = LDAB - 1

    INCA = KD1*LDAB
    KDN = min(N-1, KD)

    UPPER = UPLO == 'U'
    WANTQ = true

    if UPPER
        if KD > 1
            # Reduce to tridiagonal form, working with upper triangle
            NR = 0
            J1 = KDN + 2
            J2 = 1
            for I = 1:N-2
                # Reduce i-th row of matrix to tridiagonal form
                for K = KDN+1:-1:2
                    J1 = J1 + KDN
                    J2 = J2 + KDN
                    if NR > 0
                        # generate plane rotations to annihilate nonzero
                        # elements which have been created outside the band
                        largv!(NR, pointer(AB, 1+LDAB*(J1-2)), INCA, pointer(WORK, J1), KD1, pointer(D, J1), KD1)
                        # apply rotations from the right
                        # Dependent on the the number of diagonals either
                        # DLARTV or DROT is used
                        if NR ≥ 2*KD-1
                            for L = 1:KD-1
                                lartv!(NR, pointer(AB, L+1+LDAB*(J1-2)), INCA, pointer(AB, L+LDAB*(J1-1)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                            end
                        else
                            JEND = J1 + (NR-1)*KD1
                            for JINC = J1:KD1:JEND
                                rot!(KDM1, pointer(AB, 2+LDAB*(JINC-2)), 1, pointer(AB, 1+LDAB*(JINC-1)), 1, D[JINC], WORK[JINC])
                            end
                        end
                    end

                    if K > 2
                        if K ≤ N-I+1
                            # generate plane rotation to annihilate a(i,i+k-1)
                            # within the band
                            lartg!(AB[KD-K+3, I+K-2], AB[KD-K+2, I+K-1], refD, refWORK, TEMP)
                            D[I+K-1] = refD[]
                            WORK[I+K-1] = refWORK[]
                            AB[KD-K+3, I+K-2] = TEMP[]
                            # apply rotation from the right
                            rot!(K-3, pointer(AB, KD-K+4+LDAB*(I+K-3)), 1, pointer(AB, KD-K+3+LDAB*(I+K-2)), 1, D[I+K-1], WORK[I+K-1])
                        end
                        NR = NR + 1
                        J1 = J1 - KDN - 1
                    end

                    # apply plane rotations from both sides to diagonal
                    # blocks
                    if NR > 0
                        lar2v!(NR, pointer(AB, KD1+LDAB*(J1-2)), pointer(AB, KD1+LDAB*(J1-1)), pointer(AB, KD+LDAB*(J1-1)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                    end
                    # apply plane rotations from the left
                    if NR > 0
                        if 2*KD-1 < NR
                            # Dependent on the the number of diagonals either
                            # DLARTV or DROT is used
                            for L = 1:KD-1
                                if J2+L > N
                                    NRT = NR - 1
                                else
                                    NRT = NR
                                end
                                if NRT > 0
                                    lartv!(NRT, pointer(AB, KD-L+LDAB*(J1+L-1)), INCA, pointer(AB, KD-L+1+LDAB*(J1+L-1)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                                end
                            end
                        else
                            J1END = J1 + KD1*( NR-2 )
                            if J1END ≥ J1
                                for JIN = J1:KD1:J1END
                                    rot!(KDM1, pointer(AB, KDM1+LDAB*JIN), INCX, pointer(AB, KD+LDAB*JIN), INCX, D[JIN], WORK[JIN])
                                end
                            end
                            LEND = min( KDM1, N-J2 )
                            LAST = J1END + KD1
                            if LEND > 0
                                rot!(LEND, pointer(AB, KDM1+LDAB*LAST), INCX, pointer(AB, KD+LDAB*LAST), INCX, D[LAST], WORK[LAST])
                            end
                        end
                    end

                    if WANTQ
                        for J = J1:KD1:J2
                            push!(Q, Givens(J-1, J, D[J], WORK[J]))
                        end
                    end
                    if J2+KDN > N
                        # adjust J2 to keep within the bounds of the matrix
                        NR = NR - 1
                        J2 = J2 - KDN - 1
                    end

                    for J = J1:KD1:J2
                        # create nonzero element a(j-1,j+kd) outside the band
                        # and store it in WORK
                        WORK[J+KD] = WORK[J]*AB[1, J+KD]
                        AB[1, J+KD] = D[J]*AB[1, J+KD]
                    end
                end
            end
        end # if

        if KD > 0
            # copy off-diagonal elements to E
            for I = 1:N-1
                E[I] = AB[KD, I+1]
            end
        else
            # set E to zero if original matrix was diagonal
            for I = 1:N-1
                E[I] = ZERO
            end
        end
        # copy diagonal elements to D
        for I = 1:N
            D[I] = AB[KD1, I]
        end
    else # if UPPER
        if KD > 1
            # Reduce to tridiagonal form, working with lower triangle
            NR = 0
            J1 = KDN + 2
            J2 = 1
            for I = 1:N-2
                # Reduce i-th column of matrix to tridiagonal form
                for K = KDN+1:-1:2
                    J1 = J1 + KDN
                    J2 = J2 + KDN
                    if NR > 0
                        # generate plane rotations to annihilate nonzero
                        # elements which have been created outside the band
                        largv!(NR, pointer(AB, KD1+LDAB*(J1-KD1-1)), INCA, pointer(WORK, J1), KD1, pointer(D, J1), KD1)
                        # apply plane rotations from one side
                        # Dependent on the the number of diagonals either
                        # DLARTV or DROT is used
                        if NR > 2*KD-1
                            for L = 1:KD-1
                                lartv!(NR, pointer(AB, KD1-L+LDAB*(J1-KD1+L-1)), INCA, pointer(AB, KD1-L+1+LDAB*(J1-KD1+L-1)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                            end
                        else
                            JEND = J1 + KD1*(NR-1)
                            for JINC = J1:KD1:JEND
                                rot!(KDM1, pointer(AB, KD+LDAB*(JINC-KD1)), INCX, pointer(AB, KD1+LDAB*(JINC-KD1)), INCX, D[JINC], WORK[JINC])
                            end
                        end
                    end

                    if K > 2
                        if K ≤ N-I+1
                            # generate plane rotation to annihilate a(i+k-1,i)
                            # within the band
                            lartg!(AB[K-1, I], AB[K, I], refD, refWORK, TEMP)
                            D[I+K-1] = refD[]
                            WORK[I+K-1] = refWORK[]
                            AB[K-1, I] = TEMP[]
                            # apply rotation from the right
                            rot!(K-3, pointer(AB, K-2+LDAB*I), INCX, pointer(AB, K-1+LDAB*I), INCX, D[I+K-1], WORK[I+K-1])
                        end
                        NR = NR + 1
                        J1 = J1 - KDN - 1
                    end

                    # apply plane rotations from both sides to diagonal
                    # blocks
                    if NR > 0
                        lar2v!(NR, pointer(AB, 1+LDAB*(J1-2)), pointer(AB, 1+LDAB*(J1-1)), pointer(AB, 2+LDAB*(J1-2)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                    end
                    # apply plane rotations from the right
                    # Dependent on the the number of diagonals either
                    # DLARTV or DROT is used

                    if NR > 0
                        if NR > 2*KD-1
                            for L = 1:KD-1
                                if J2+L > N
                                    NRT = NR - 1
                                else
                                    NRT = NR
                                end
                                if NRT > 0
                                    lartv!(NRT, pointer(AB, L+2+LDAB*(J1-2)), INCA, pointer(AB, L+1+LDAB*(J1-1)), INCA, pointer(D, J1), pointer(WORK, J1), KD1)
                                end
                            end
                        else
                            J1END = J1 + KD1*(NR-2)
                            if J1END ≥ J1
                                for J1INC = J1:KD1:J1END
                                    rot!(KDM1, pointer(AB, 3+LDAB*(J1INC-2)), 1, pointer(AB, 2+LDAB*(J1INC-1)), 1, D[J1INC], WORK[J1INC])
                                end
                            end
                            LEND = min(KDM1, N-J2)
                            LAST = J1END + KD1
                            if LEND > 0
                                rot!(LEND, pointer(AB, 3+LDAB*(LAST-2)), 1, pointer(AB, 2+LDAB*(LAST-1)), 1, D[LAST], WORK[LAST])
                            end
                        end
                    end

                    if WANTQ
                        for J = J1:KD1:J2
                            push!(Q, Givens(J-1, J, D[J], WORK[J]))
                        end
                    end

                    if J2+KDN > N
                        # adjust J2 to keep within the bounds of the matrix
                        NR = NR - 1
                        J2 = J2 - KDN - 1
                    end

                    for J = J1:KD1:J2
                        # create nonzero element a(j+kd,j-1) outside the band
                        # and store it in WORK
                        WORK[J+KD] = WORK[J]*AB[KD1, J]
                        AB[KD1, J] = D[J]*AB[KD1, J]
                    end
                end
            end
        end # if
        if KD > 0
            # copy off-diagonal elements to E
            for I = 1:N-1
                E[I] = AB[2, I]
            end
        else
            # set E to zero if original matrix was diagonal
            for I = 1:N-1
                E[I] = ZERO
            end
        end
        # copy diagonal elements to D
        for I = 1:N
            D[I] = AB[1, I]
        end
    end # if UPPER
end
