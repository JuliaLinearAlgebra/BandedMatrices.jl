# V = G Q
struct BandedEigenvectors{T} <: AbstractMatrix{T}
    G::Vector{Givens{T}}
    Q::Matrix{T}
end

size(B::BandedEigenvectors) = size(B.Q)
getindex(B::BandedEigenvectors, i, j) = Matrix(B)[i, j]

# V = S⁻¹ Q W
struct BandedGeneralizedEigenvectors{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    S::SplitCholesky{T,M}
    Q::Vector{Givens{T}}
    W::BandedEigenvectors{T}
end

size(B::BandedGeneralizedEigenvectors) = size(B.W)
getindex(B::BandedGeneralizedEigenvectors, i, j) = Matrix(B)[i, j]

convert(::Type{Eigen{T, T, Matrix{T}, Vector{T}}}, F::Eigen{T, T, BandedEigenvectors{T}, Vector{T}}) where T = Eigen(F.values, Matrix(F.vectors))
convert(::Type{GeneralizedEigen{T, T, Matrix{T}, Vector{T}}}, F::GeneralizedEigen{T, T, BandedGeneralizedEigenvectors{T}, Vector{T}}) where T = GeneralizedEigen(F.values, Matrix(F.vectors))

compress(F::Eigen{T, T, BandedEigenvectors{T}, Vector{T}}) where T = convert(Eigen{T, T, Matrix{T}, Vector{T}}, F)
compress(F::GeneralizedEigen{T, T, BandedGeneralizedEigenvectors{T}, Vector{T}}) where T = convert(GeneralizedEigen{T, T, Matrix{T}, Vector{T}}, F)

eigen(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigen!(copy(A))
eigen(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T <: Real = eigen!(copy(A), copy(B))

function eigen!(A::Symmetric{T,<:BandedMatrix{T}}) where T <: Real
    N = size(A, 1)
    KD = bandwidth(A)
    D = Vector{T}(undef, N)
    E = Vector{T}(undef, N-1)
    G = Vector{Givens{T}}(undef, 0)
    WORK = Vector{T}(undef, N)
    AB = symbandeddata(A)
    sbtrd!('V', A.uplo, N, KD, AB, D, E, G, WORK)
    Λ, Q = eigen(SymTridiagonal(D, E))
    Eigen(Λ, BandedEigenvectors(G, Q))
end

function eigen!(A::Symmetric{T,<:BandedMatrix{T}}, B::Symmetric{T,<:BandedMatrix{T}}) where T <: Real
    S = splitcholesky!(B)
    N = size(A, 1)
    KA = bandwidth(A)
    KB = bandwidth(B)
    Q = Vector{Givens{T}}(undef, 0)
    WORK = Vector{T}(undef, 2*N)
    AB = symbandeddata(A)
    BB = symbandeddata(B)
    sbgst!('V', A.uplo, N, KA, KB, AB, BB, Q, WORK)
    Λ, W = eigen!(A)
    GeneralizedEigen(Λ, BandedGeneralizedEigenvectors(S, Q, W))
end

function Matrix(B::BandedEigenvectors)
    Q = copy(B.Q)
    G = B.G
    for k in length(G):-1:1
        lmul!(G[k], Q)
    end
    return Q
end

function Matrix(B::BandedGeneralizedEigenvectors)
    V = Matrix(B.W)
    Q = B.Q
    for k in length(Q):-1:1
        lmul!(Q[k], V)
    end
    ldiv!(B.S, V)
    return V
end


function compress!(F::Eigen{T, T, BandedEigenvectors{T}, Vector{T}}) where T
    Q = F.vectors.Q
    G = F.vectors.G
    for k in length(G):-1:1
        lmul!(G[k], Q)
        pop!(G)
    end
    F
end

function mul!(y::Array{T,N}, B::BandedEigenvectors{T}, x::Array{T,N}) where {T,N}
    mul!(y, B.Q, x)
    G = B.G
    for k in length(G):-1:1
        lmul!(G[k], y)
    end
    y
end

function mul!(y::Array{T,N}, B::Adjoint{T,BandedEigenvectors{T}}, x::Array{T,N}) where {T,N}
    Q = B.parent.Q
    G = B.parent.G
    if length(G) > 0
        z = copy(x)
        for k in 1:length(G)
            lmul!(G[k]', z)
        end
        mul!(y, Q', z)
    else
        mul!(y, Q', x)
    end
    y
end

function mul!(y::Array{T,N}, B::Transpose{T,BandedEigenvectors{T}}, x::Array{T,N}) where {T,N}
    Q = B.parent.Q
    G = B.parent.G
    if length(G) > 0
        z = copy(x)
        for k in 1:length(G)
            lmul!(G[k]', z)
        end
        mul!(y, transpose(Q), z)
    else
        mul!(y, transpose(Q), x)
    end
    y
end

function mul!(y::Array{T,N}, B::BandedGeneralizedEigenvectors{T}, x::Array{T,N}) where {T,N}
    mul!(y, B.W, x)
    Q = B.Q
    for k in length(Q):-1:1
        lmul!(Q[k], y)
    end
    ldiv!(B.S, y)
end

function mul!(y::Array{T,N}, B::Adjoint{T,BandedGeneralizedEigenvectors{T,M}}, x::Array{T,N}) where {T,M,N}
    z = copy(x)
    ldiv!(B.parent.S', z)
    Q = B.parent.Q
    for k in 1:length(Q)
        lmul!(Q[k]', z)
    end
    mul!(y, B.parent.W', z)
end

function ldiv!(y::Array{T,N}, B::BandedGeneralizedEigenvectors{T}, x::Array{T,N}) where {T,N}
    z = copy(x)
    lmul!(B.S, z)
    Q = B.Q
    for k in 1:length(Q)
        lmul!(Q[k]', z)
    end
    mul!(y, B.W', z)
end

function mul!(y::Array{T,N}, x::Array{T,N}, B::BandedEigenvectors{T}) where {T,N}
    x .= x'
    mul!(y, B', x)
    x .= x'
    y .= y'
end

function mul!(y::Array{T,N}, x::Array{T,N}, B::BandedGeneralizedEigenvectors{T}) where {T,N}
    x .= x'
    mul!(y, B', x)
    x .= x'
    y .= y'
end



#
# The following is a Julia translation of *SBTRD.f in LAPACK that allows
# extraction of the Givens rotations.
#
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

    # In Julia, pointers to entries of a SubArray ignore stride.
    # For example, if `A = view(Matrix{Float64}(I, 10, 10), 1:2:9, 1:2:9)`,
    # then `unsafe_load(pointer(A, 3+5*2)) == 1`, even if `stride(A, 2) == 20.`
    # Beyond the use of `LDAB` in Julia pointers, there are two uses of the
    # actual (Fortran) `LDABF` for increment-setting in the LAPACK calls.
    LDAB = size(AB, 1)
    LDABF = max(1, stride(AB, 2))

    ZERO = zero(T)
    TEMP1 = Ref{T}()
    TEMP2 = Ref{T}()
    TEMP3 = Ref{T}()
    KD1 = KD + 1
    KDM1 = KD - 1
    INCX = LDABF - 1

    INCA = KD1*LDABF
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
                            lartg!(AB[KD-K+3, I+K-2], AB[KD-K+2, I+K-1], TEMP1, TEMP2, TEMP3)
                            D[I+K-1] = TEMP1[]
                            WORK[I+K-1] = TEMP2[]
                            AB[KD-K+3, I+K-2] = TEMP3[]
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
                            push!(Q, Givens(J-1, J, D[J], -WORK[J]))
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
                            lartg!(AB[K-1, I], AB[K, I], TEMP1, TEMP2, TEMP3)
                            D[I+K-1] = TEMP1[]
                            WORK[I+K-1] = TEMP2[]
                            AB[K-1, I] = TEMP3[]
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
                            push!(Q, Givens(J-1, J, D[J], -WORK[J]))
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


#
# The following is a Julia translation of *SBGST.f in LAPACK that allows
# extraction of the Givens rotations.
#
#  Definition:
#  ===========
#
#       SUBROUTINE DSBGST( VECT, UPLO, N, KA, KB, AB, LDAB, BB, LDBB, X,
#                          LDX, WORK, INFO )
#
#       .. Scalar Arguments ..
#       CHARACTER          UPLO, VECT
#       INTEGER            INFO, KA, KB, LDAB, LDBB, LDX, N
#       ..
#       .. Array Arguments ..
#       DOUBLE PRECISION   AB( LDAB, * ), BB( LDBB, * ), WORK( * ),
#      $                    X( LDX, * )
#       ..
function sbgst!(VECT::Char, UPLO::Char,
                N::Int, KA::Int, KB::Int, AB::AbstractMatrix{T},
                BB::AbstractMatrix{T}, X::Vector{Givens{T}},
                WORK::AbstractVector{T}) where T
    require_one_based_indexing(AB)
    require_one_based_indexing(BB)
    chkstride1(AB)
    chkstride1(BB)
    chkuplo(UPLO)
    chkvect(VECT)
    size(AB,2) == size(BB,2) == N || throw(ArgumentError("Matrices must be square"))
    size(AB,1) < KA+1 && throw(ArgumentError("Not enough bands in AB"))
    size(BB,1) < KB+1 && throw(ArgumentError("Not enough bands in BB"))

    # In Julia, pointers to entries of a SubArray ignore stride.
    # For example, if `A = view(Matrix{Float64}(I, 10, 10), 1:2:9, 1:2:9)`,
    # then `unsafe_load(pointer(A, 3+5*2)) == 1`, even if `stride(A, 2) == 20.`
    # Beyond the use of `LDAB` in Julia pointers, there are two uses of the
    # actual (Fortran) `LDABF` for increment-setting in the LAPACK calls.
    LDAB = size(AB, 1)
    LDABF = max(1, stride(AB, 2))

    ZERO = zero(T)
    TEMP1 = Ref{T}()
    TEMP2 = Ref{T}()
    TEMP3 = Ref{T}()
    KA1 = KA + 1
    KB1 = KB + 1
    INCA = LDABF*KA1
    M = ( N+KB ) ÷ 2

    UPPER = UPLO == 'U'
    WANTX = true

    UPDATE = true
    I = N + 1
    @label L10
    if UPDATE
        I = I - 1
        KBT = min( KB, I-1 )
        I0 = I - 1
        I1 = min( N, I+KA )
        I2 = I - KBT + KA1
        if I < M+1
            UPDATE = false
            I = I + 1
            I0 = M
            if KA == 0
                @goto L480
            end
            @goto L10
        end
    else
        I = I + KA
        if I > N-1
            @goto L480
        end
    end

    if UPPER
        # Transform A, working with the upper triangle
        if UPDATE
            # Form inv(S(i))**T * A * inv(S(i))
            BII = BB[KB1, I]
            for J = I:I1
                AB[I-J+KA1, J] = AB[I-J+KA1, J] / BII
            end
            for J = max(1, I-KA):I
                AB[J-I+KA1, I] = AB[J-I+KA1, I] / BII
            end
            for K = I-KBT:I-1
                for J = I-KBT:K
                    AB[J-K+KA1, K] = AB[J-K+KA1, K] - BB[J-I+KB1, I]*AB[K-I+KA1, I] - BB[K-I+KB1, I]*AB[J-I+KA1, I] + AB[KA1, I]*BB[J-I+KB1, I]*BB[K-I+KB1, I]
                end
                for J = max(1, I-KA):I-KBT-1
                    AB[J-K+KA1, K] = AB[J-K+KA1, K] - BB[K-I+KB1, I]*AB[J-I+KA1, I]
                end
            end
            for J = I:I1
                for K = max(J-KA, I-KBT):I-1
                    AB[K-J+KA1, J] = AB[K-J+KA1, J] - BB[K-I+KB1, I]*AB[I-J+KA1, J]
                end
            end
            # store a(i,i1) in RA1 for use in next loop over K
            RA1 = AB[I-I1+KA1, I1]
        end
        # Generate and apply vectors of rotations to chase all the
        # existing bulges KA positions down toward the bottom of the
        # band
        for K = 1:KB-1
            if UPDATE
                # Determine the rotations which would annihilate the bulge
                # which has in theory just been created
                if I-K+KA < N && I-K > 1
                    # generate rotation to annihilate a(i,i-k+ka+1)
                    lartg!(AB[K+1, I-K+KA], RA1, TEMP1, TEMP2, TEMP3)
                    WORK[N+I-K+KA-M] = TEMP1[]
                    WORK[I-K+KA-M] = TEMP2[]
                    RA = TEMP3[]
                    # create nonzero element a(i-k,i-k+ka+1) outside the
                    # band and store it in WORK(i-k)
                    t = -BB[KB1-K, I]*RA1
                    WORK[I-K] = WORK[N+I-K+KA-M]*t - WORK[I-K+KA-M]*AB[1, I-K+KA]
                    AB[1, I-K+KA] = WORK[I-K+KA-M]*t + WORK[N+I-K+KA-M]*AB[1, I-K+KA]
                    RA1 = RA
                end
            end
            J2 = I - K - 1 + max(1, K-I0+2)*KA1
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            if UPDATE
                J2T = max( J2, I+2*KA-K+1 )
            else
                J2T = J2
            end
            NRT = ( N-J2T+KA ) ÷ KA1
            for J = J2T:KA1:J1
                # create nonzero element a(j-ka,j+1) outside the band
                # and store it in WORK(j-m)
                WORK[J-M] = WORK[J-M]*AB[1, J+1]
                AB[1, J+1] = WORK[N+J-M]*AB[1, J+1]
            end
            # generate rotations in 1st set to annihilate elements which
            # have been created outside the band
            if NRT > 0
                largv!(NRT, pointer(AB, 1+LDAB*(J2T-1)), INCA, pointer(WORK, J2T-M), KA1, pointer(WORK, N+J2T-M), KA1)
            end
            if NR > 0
                # apply rotations in 1st set from the right
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, KA1-L+LDAB*(J2-1)), INCA, pointer(AB, KA-L+LDAB*J2), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
                # apply rotations in 1st set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, KA1+LDAB*(J2-1)), pointer(AB, KA1+LDAB*J2), pointer(AB, KA+LDAB*J2), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
            end
            # start applying rotations in 1st set from the left
            for L = KA-1:-1:KB-K+1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J2+KA1-L-1)), INCA, pointer(AB, L+1+LDAB*(J2+KA1-L-1)), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 1st set
                for J = J2:KA1:J1
                    push!(X, Givens(J, J+1, WORK[N+J-M], -WORK[J-M]))
                end
            end
        end

        if UPDATE
            if I2 ≤ N && KBT > 0
                # create nonzero element a(i-kbt,i-kbt+ka+1) outside the
                # band and store it in WORK(i-kbt)
                WORK[I-KBT] = -BB[KB1-KBT, I]*RA1
            end
        end

        for K = KB:-1:1
            if UPDATE
                J2 = I - K - 1 + max( 2, K-I0+1 )*KA1
            else
                J2 = I - K - 1 + max( 1, K-I0+1 )*KA1
            end
            # finish applying rotations in 2nd set from the left
            for L = KB-K:-1:1
                NRT = ( N-J2+KA+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J2-L)), INCA, pointer(AB, L+1+LDAB*(J2-L)), INCA, pointer(WORK, N+J2-KA), pointer(WORK, J2-KA), KA1)
                end
            end
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            for J = J1:-KA1:J2
                WORK[J] = WORK[J-KA]
                WORK[N+J] = WORK[N+J-KA]
            end
            for J = J2:KA1:J1
                # create nonzero element a(j-ka,j+1) outside the band
                # and store it in WORK(j)
                WORK[J] = WORK[J]*AB[1, J+1]
                AB[1, J+1] = WORK[N+J]*AB[1, J+1]
            end
            if UPDATE
                if I-K < N-KA && K ≤ KBT
                    WORK[I-K+KA] = WORK[I-K]
                end
            end
        end

        for K = KB:-1:1
            J2 = I - K - 1 + max( 1, K-I0+1 )*KA1
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            if NR > 0
                # generate rotations in 2nd set to annihilate elements
                # which have been created outside the band
                largv!(NR, pointer(AB, 1+LDAB*(J2-1)), INCA, pointer(WORK, J2), KA1, pointer(WORK, N+J2), KA1)
                # apply rotations in 2nd set from the right
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, KA1-L+LDAB*(J2-1)), INCA, pointer(AB, KA-L+LDAB*J2), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
                end
                # apply rotations in 2nd set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, KA1+LDAB*(J2-1)), pointer(AB, KA1+LDAB*J2), pointer(AB, KA+LDAB*J2), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
            end
            # start applying rotations in 2nd set from the left
            for L = KA-1:-1:KB-K+1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J2+KA1-L-1)), INCA, pointer(AB, L+1+LDAB*(J2+KA1-L-1)), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 2nd set
                for J = J2:KA1:J1
                    push!(X, Givens(J, J+1, WORK[N+J], -WORK[J]))
                end
            end
        end

        for K = 1:KB-1
            J2 = I - K - 1 + max( 1, K-I0+2 )*KA1
            for L = KB-K:-1:1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J2+KA1-L-1)), INCA, pointer(AB, L+1+LDAB*(J2+KA1-L-1)), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
            end
        end

        if KB > 1
            for J = N-1:-1:I-KB+2*KA+1
                WORK[N+J-M] = WORK[N+J-KA-M]
                WORK[J-M] = WORK[J-KA-M]
            end
        end
    else
        # Transform A, working with the lower triangle
        if UPDATE
            # Form inv(S(i))**T * A * inv(S(i))
            BII = BB[1, I]
            for J = I:I1
                AB[J-I+1, I] = AB[J-I+1, I] / BII
            end
            for J = max(1,I-KA):I
                AB[I-J+1, J] = AB[I-J+1, J] / BII
            end
            for K = I-KBT:I-1
                for J = I-KBT:K
                    AB[K-J+1,J] = AB[K-J+1, J] - BB[I-J+1, J]*AB[I-K+1, K] - BB[I-K+1, K]*AB[I-J+1, J] + AB[1, I]*BB[I-J+1, J]*BB[I-K+1, K]
                end
                for J = max(1, I-KA):I-KBT-1
                    AB[K-J+1, J] = AB[K-J+1, J] - BB[I-K+1, K]*AB[I-J+1, J]
                end
            end
            for J = I:I1
                for K = max(J-KA, I-KBT):I-1
                    AB[J-K+1, K] = AB[J-K+1, K] - BB[I-K+1, K]*AB[J-I+1, I]
                end
            end
            # store a(i1,i) in RA1 for use in next loop over K
            RA1 = AB[I1-I+1, I]
        end
        # Generate and apply vectors of rotations to chase all the
        # existing bulges KA positions down toward the bottom of the
        # band
        for K = 1:KB-1
            if UPDATE
                # Determine the rotations which would annihilate the bulge
                # which has in theory just been created
                if I-K+KA < N && I-K > 1
                    # generate rotation to annihilate a(i-k+ka+1,i)
                    lartg!(AB[KA1-K, I], RA1, TEMP1, TEMP2, TEMP3)
                    WORK[N+I-K+KA-M] = TEMP1[]
                    WORK[I-K+KA-M] = TEMP2[]
                    RA = TEMP3[]
                    t = -BB[K+1, I-K]*RA1
                    WORK[I-K] = WORK[N+I-K+KA-M]*t - WORK[I-K+KA-M]*AB[KA1, I-K]
                    AB[KA1, I-K] = WORK[I-K+KA-M]*t + WORK[N+I-K+KA-M]*AB[KA1, I-K]
                    RA1 = RA
                end
            end
            J2 = I - K - 1 + max( 1, K-I0+2 )*KA1
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            if UPDATE
                J2T = max( J2, I+2*KA-K+1 )
            else
                J2T = J2
            end
            NRT = ( N-J2T+KA ) ÷ KA1
            for J = J2T:KA1:J1
                # create nonzero element a(j+1,j-ka) outside the band
                # and store it in WORK(j-m)
                WORK[J-M] = WORK[J-M]*AB[KA1, J-KA+1]
                AB[KA1, J-KA+1] = WORK[N+J-M]*AB[KA1, J-KA+1]
            end
            # generate rotations in 1st set to annihilate elements which
            # have been created outside the band
            if NRT > 0
                largv!(NRT, pointer(AB, KA1+LDAB*(J2T-KA-1)), INCA, pointer(WORK, J2T-M), KA1, pointer(WORK, N+J2T-M), KA1)
            end
            if NR > 0
                # apply rotations in 1st set from the left
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, L+1+LDAB*(J2-L-1)), INCA, pointer(AB, L+2+LDAB*(J2-L-1)), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
                # apply rotations in 1st set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, 1+LDAB*(J2-1)), pointer(AB, 1+LDAB*J2), pointer(AB, 2+LDAB*(J2-1)), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
            end
            # start applying rotations in 1st set from the right
            for L = KA-1:-1:KB-K+1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J2-1)), INCA, pointer(AB, KA1-L+LDAB*J2), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 1st set
                for J = J2:KA1:J1
                    push!(X, Givens(J, J+1, WORK[N+J-M], -WORK[J-M]))
                end
            end
        end

        if UPDATE
            if I2 ≤ N && KBT > 0
                # create nonzero element a(i-kbt+ka+1,i-kbt) outside the
                # band and store it in WORK(i-kbt)
                WORK[I-KBT] = -BB[KBT+1, I-KBT]*RA1
            end
        end

        for K = KB:-1:1
            if UPDATE
                J2 = I - K - 1 + max( 2, K-I0+1 )*KA1
            else
                J2 = I - K - 1 + max( 1, K-I0+1 )*KA1
            end
            # finish applying rotations in 2nd set from the right
            for L = KB-K:-1:1
                NRT = ( N-J2+KA+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J2-KA-1)), INCA, pointer(AB, KA1-L+LDAB*(J2-KA)), INCA, pointer(WORK, N+J2-KA), pointer(WORK, J2-KA), KA1)
                end
            end
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            for J = J1:-KA1:J2
                WORK[J] = WORK[J-KA]
                WORK[N+J] = WORK[N+J-KA]
            end
            for J = J2:KA1:J1
                # create nonzero element a(j+1,j-ka) outside the band
                # and store it in WORK(j)
                WORK[J] = WORK[J]*AB[KA1, J-KA+1]
                AB[KA1, J-KA+1] = WORK[N+J]*AB[KA1, J-KA+1]
            end
            if UPDATE
                if I-K < N-KA && K ≤ KBT
                    WORK[I-K+KA] = WORK[I-K]
                end
            end
        end

        for K = KB:-1:1
            J2 = I - K - 1 + max( 1, K-I0+1 )*KA1
            NR = ( N-J2+KA ) ÷ KA1
            J1 = J2 + ( NR-1 )*KA1
            if NR > 0
                # generate rotations in 2nd set to annihilate elements
                # which have been created outside the band
                largv!(NR, pointer(AB, KA1+LDAB*(J2-KA-1)), INCA, pointer(WORK, J2), KA1, pointer(WORK, N+J2), KA1)
                # apply rotations in 2nd set from the left
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, L+1+LDAB*(J2-L-1)), INCA, pointer(AB, L+2+LDAB*(J2-L-1)), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
                end
                # apply rotations in 2nd set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, 1+LDAB*(J2-1)), pointer(AB, 1+LDAB*J2), pointer(AB, 2+LDAB*(J2-1)), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
            end
            # start applying rotations in 2nd set from the right
            for L = KA-1:-1:KB-K+1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J2-1)), INCA, pointer(AB, KA1-L+LDAB*J2), INCA, pointer(WORK, N+J2), pointer(WORK, J2), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 2nd set
                for J = J2:KA1:J1
                    push!(X, Givens(J, J+1, WORK[N+J], -WORK[J]))
                end
            end
        end

        for K = 1:KB-1
            J2 = I - K - 1 + max( 1, K-I0+2 )*KA1
            # finish applying rotations in 1st set from the right
            for L = KB-K:-1:1
                NRT = ( N-J2+L ) ÷ KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J2-1)), INCA, pointer(AB, KA1-L+LDAB*J2), INCA, pointer(WORK, N+J2-M), pointer(WORK, J2-M), KA1)
                end
            end
        end

        if KB > 1
            for J = N-1:-1:I-KB+2*KA+1
                WORK[N+J-M] = WORK[N+J-KA-M]
                WORK[J-M] = WORK[J-KA-M]
            end
        end
    end
    @goto L10
    @label L480

    UPDATE = true
    I = 0
    @label L490
    if UPDATE
        I = I + 1
        KBT = min( KB, M-I )
        I0 = I + 1
        I1 = max( 1, I-KA )
        I2 = I + KBT - KA1
        if I > M
            UPDATE = false
            I = I - 1
            I0 = M + 1
            if KA == 0
                return
            end
            @goto L490
        end
    else
        I = I - KA
        if I < 2
            return
        end
    end

    if I < M - KBT
        NX = M
    else
        NX = N
    end

    if UPPER
        # Transform A, working with the upper triangle
        if UPDATE
            # Form inv(S(i))**T * A * inv(S(i))
            BII = BB[KB1, I]
            for J = I1:I
                AB[J-I+KA1, I] = AB[J-I+KA1, I] / BII
            end
            for J = I:min(N, I+KA)
                AB[I-J+KA1, J] = AB[I-J+KA1, J] / BII
            end
            for K = I+1:I+KBT
                for J = K:I+KBT
                    AB[K-J+KA1, J] = AB[K-J+KA1, J] - BB[I-J+KB1, J]*AB[I-K+KA1, K] - BB[I-K+KB1, K]*AB[I-J+KA1, J] + AB[KA1, I]*BB[I-J+KB1, J]*BB[I-K+KB1, K]
                end
                for J = I+KBT+1:min(N, I+KA)
                    AB[K-J+KA1, J] = AB[K-J+KA1, J] - BB[I-K+KB1, K]*AB[I-J+KA1, J]
                end
            end
            for J = I1:I
                for K = I+1:min(J+KA, I+KBT)
                    AB[J-K+KA1, K] = AB[J-K+KA1, K] - BB[I-K+KB1, K]*AB[J-I+KA1, I]
                end
            end
            # store a(i1,i) in RA1 for use in next loop over K
            RA1 = AB[I1-I+KA1, I]
        end

        for K = 1:KB-1
            if UPDATE
                # Determine the rotations which would annihilate the bulge
                # which has in theory just been created
                if I+K-KA1 > 0 && I+K < M
                    # generate rotation to annihilate a(i+k-ka-1,i)
                    lartg!(AB[K+1, I], RA1, TEMP1, TEMP2, TEMP3)
                    WORK[N+I+K-KA] = TEMP1[]
                    WORK[I+K-KA] = TEMP2[]
                    RA = TEMP3[]
                    # create nonzero element a(i+k-ka-1,i+k) outside the
                    # band and store it in WORK(m-kb+i+k)
                    t = -BB[KB1-K, I+K]*RA1
                    WORK[M-KB+I+K] = WORK[N+I+K-KA]*t - WORK[I+K-KA]*AB[1, I+K]
                    AB[1, I+K] = WORK[I+K-KA]*t + WORK[N+I+K-KA]*AB[1, I+K]
                    RA1 = RA
                end
            end
            J2 = I + K + 1 - max( 1, K+I0-M+1 )*KA1
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            if UPDATE
                J2T = min(J2, I-2*KA+K-1)
            else
                J2T = J2
            end
            NRT = ( J2T+KA-1 ) ÷ KA1
            for J = J1:KA1:J2T
                # create nonzero element a(j-1,j+ka) outside the band
                # and store it in WORK(j)
                WORK[J] = WORK[J]*AB[1, J+KA-1]
                AB[1, J+KA-1] = WORK[N+J]*AB[1, J+KA-1]
            end
            # generate rotations in 1st set to annihilate elements which
            # have been created outside the band

            if NRT > 0
                largv!(NRT, pointer(AB, 1+LDAB*(J1+KA-1)), INCA, pointer(WORK, J1), KA1, pointer(WORK, N+J1), KA1)
            end
            if NR > 0
                # apply rotations in 1st set from the left
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, KA1-L+LDAB*(J1+L-1)), INCA, pointer(AB, KA-L+LDAB*(J1+L-1)), INCA, pointer(WORK, N+J1), pointer(WORK, J1), KA1)
                end
                # apply rotations in 1st set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, KA1+LDAB*(J1-1)), pointer(AB, KA1+LDAB*(J1-2)), pointer(AB, KA+LDAB*(J1-1)), INCA, pointer(WORK, N+J1), pointer(WORK, J1), KA1)
            end
            # start applying rotations in 1st set from the right
            for L = KA-1:-1:KB-K+1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J1T-1)), INCA, pointer(AB, L+1+LDAB*(J1T-2)), INCA, pointer(WORK, N+J1T), pointer(WORK, J1T), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 1st set
                for J = J1:KA1:J2
                    push!(X, Givens(J, J-1, WORK[N+J], -WORK[J]))
                end
            end
        end

        if UPDATE
            if I2 > 0 && KBT > 0
                # create nonzero element a(i+kbt-ka-1,i+kbt) outside the
                # band and store it in WORK(m-kb+i+kbt)
                WORK[M-KB+I+KBT] = -BB[KB1-KBT, I+KBT]*RA1
            end
        end

        for K = KB:-1:1
            if UPDATE
                J2 = I + K + 1 - max( 2, K+I0-M )*KA1
            else
                J2 = I + K + 1 - max( 1, K+I0-M )*KA1
            end
            # finish applying rotations in 2nd set from the right
            for L = KB-K:-1:1
                NRT = ( J2+KA+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J1T+KA-1)), INCA, pointer(AB, L+1+LDAB*(J1T+KA-2)), INCA, pointer(WORK, N+M-KB+J1T+KA), pointer(WORK, M-KB+J1T+KA), KA1)
                end
            end
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            for J = J1:KA1:J2
                WORK[M-KB+J] = WORK[M-KB+J+KA]
                WORK[N+M-KB+J] = WORK[N+M-KB+J+KA]
            end
            for J = J1:KA1:J2
                # create nonzero element a(j-1,j+ka) outside the band
                # and store it in WORK(m-kb+j)
                WORK[M-KB+J] = WORK[M-KB+J]*AB[1, J+KA-1]
                AB[1, J+KA-1] = WORK[N+M-KB+J]*AB[1, J+KA-1]
            end
            if UPDATE
                if I+K > KA1 && K ≤ KBT
                    WORK[M-KB+I+K-KA] = WORK[M-KB+I+K]
                end
            end
        end

        for K = KB:-1:1
            J2 = I + K + 1 - max( 1, K+I0-M )*KA1
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            if NR > 0
                # generate rotations in 2nd set to annihilate elements
                # which have been created outside the band
                largv!(NR, pointer(AB, 1+LDAB*(J1+KA-1)), INCA, pointer(WORK, M-KB+J1), KA1, pointer(WORK, N+M-KB+J1), KA1)
                # apply rotations in 2nd set from the left
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, KA1-L+LDAB*(J1+L-1)), INCA, pointer(AB, KA-L+LDAB*(J1+L-1)), INCA, pointer(WORK, N+M-KB+J1), pointer(WORK, M-KB+J1), KA1)
                end
                # apply rotations in 2nd set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, KA1+LDAB*(J1-1)), pointer(AB, KA1+LDAB*(J1-2)), pointer(AB, KA+LDAB*(J1-1)), INCA, pointer(WORK, N+M-KB+J1), pointer(WORK, M-KB+J1), KA1)
            end
            # start applying rotations in 2nd set from the right
            for L = KA-1:-1:KB-K+1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J1T-1)), INCA, pointer(AB, L+1+LDAB*(J1T-2)), INCA, pointer(WORK, N+M-KB+J1T), pointer(WORK, M-KB+J1T), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 2nd set
                for J = J1:KA1:J2
                    push!(X, Givens(J, J-1, WORK[N+M-KB+J], -WORK[M-KB+J]))
                end
            end
        end

        for K = 1:KB-1
            J2 = I + K + 1 - max( 1, K+I0-M+1 )*KA1
            # finish applying rotations in 1st set from the right
            for L = KB-K:-1:1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, L+LDAB*(J1T-1)), INCA, pointer(AB, L+1+LDAB*(J1T-2)), INCA, pointer(WORK, N+J1T), pointer(WORK, J1T), KA1)
                end
            end
        end

        if KB > 1
            for J = 2:min(I+KB, M)-2*KA-1
                WORK[N+J] = WORK[N+J+KA]
                WORK[J] = WORK[J+KA]
            end
        end
    else
        # Transform A, working with the lower triangle
        if UPDATE
            # Form inv(S(i))**T * A * inv(S(i))
            BII = BB[1, I]
            for J = I1:I
                AB[I-J+1, J] = AB[I-J+1, J] / BII
            end
            for J = I:min(N, I+KA)
                AB[J-I+1, I] = AB[J-I+1, I] / BII
            end
            for K = I+1:I+KBT
                for J = K:I+KBT
                    AB[J-K+1, K] = AB[J-K+1, K] - BB[J-I+1, I]*AB[K-I+1, I] - BB[K-I+1, I]*AB[J-I+1, I] + AB[1, I]*BB[J-I+1, I]*BB[K-I+1, I]
                end
                for J = I+KBT+1:min(N,I+KA)
                    AB[J-K+1, K] = AB[J-K+1, K] - BB[K-I+1, I]*AB[J-I+1, I]
                end
            end
            for J = I1:I
                for K = I+1:min(J+KA, I+KBT)
                    AB[K-J+1, J] = AB[K-J+1, J] - BB[K-I+1, I]*AB[I-J+1, J]
                end
            end
            # store a(i,i1) in RA1 for use in next loop over K
            RA1 = AB[I-I1+1, I1]
        end
        # Generate and apply vectors of rotations to chase all the
        # existing bulges KA positions up toward the top of the band
        for K = 1:KB-1
            if UPDATE
                # Determine the rotations which would annihilate the bulge
                # which has in theory just been created
                if I+K-KA1 > 0 && I+K < M
                    # generate rotation to annihilate a(i,i+k-ka-1)
                    lartg!(AB[KA1-K, I+K-KA], RA1, TEMP1, TEMP2, TEMP3)
                    WORK[N+I+K-KA] = TEMP1[]
                    WORK[I+K-KA] = TEMP2[]
                    RA = TEMP3[]
                    # create nonzero element a(i+k,i+k-ka-1) outside the
                    # band and store it in WORK(m-kb+i+k)
                    t = -BB[K+1, I]*RA1
                    WORK[M-KB+I+K] = WORK[N+I+K-KA]*t - WORK[I+K-KA]*AB[KA1, I+K-KA]
                    AB[KA1, I+K-KA] = WORK[I+K-KA]*t + WORK[N+I+K-KA]*AB[KA1, I+K-KA]
                    RA1 = RA
                end
            end
            J2 = I + K + 1 - max( 1, K+I0-M+1 )*KA1
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            if UPDATE
                J2T = min( J2, I-2*KA+K-1 )
            else
                J2T = J2
            end
            NRT = ( J2T+KA-1 ) ÷ KA1
            for J = J1:KA1:J2T
                # create nonzero element a(j+ka,j-1) outside the band
                # and store it in WORK(j)
                WORK[J] = WORK[J]*AB[KA1, J-1]
                AB[KA1, J-1] = WORK[N+J]*AB[KA1, J-1]
            end
            # generate rotations in 1st set to annihilate elements which
            # have been created outside the band
            if NRT > 0
                largv!(NRT, pointer(AB, KA1+LDAB*(J1-1)), INCA, pointer(WORK, J1), KA1, pointer(WORK, N+J1), KA1)
            end
            if NR > 0
                # apply rotations in 1st set from the right
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, L+1+LDAB*(J1-1)), INCA, pointer(AB, L+2+LDAB*(J1-2)), INCA, pointer(WORK, N+J1), pointer(WORK, J1), KA1)
                end
                # apply rotations in 1st set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, 1+LDAB*(J1-1)), pointer(AB, 1+LDAB*(J1-2)), pointer(AB, 2+LDAB*(J1-2)), INCA, pointer(WORK, N+J1), pointer(WORK, J1), KA1)
            end
            # start applying rotations in 1st set from the left
            for L = KA-1:-1:KB-K+1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J1T-KA1+L-1)), INCA, pointer(AB, KA1-L+LDAB*(J1T-KA1+L-1)), INCA, pointer(WORK, N+J1T), pointer(WORK, J1T), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 1st set
                for J = J1:KA1:J2
                    push!(X, Givens(J, J-1, WORK[N+J], -WORK[J]))
                end
            end
        end

        if UPDATE
            if I2 > 0 && KBT > 0
                # create nonzero element a(i+kbt,i+kbt-ka-1) outside the
                # band and store it in WORK(m-kb+i+kbt)
                WORK[M-KB+I+KBT] = -BB[KBT+1, I]*RA1
            end
        end

        for K = KB:-1:1
            if UPDATE
                J2 = I + K + 1 - max( 2, K+I0-M )*KA1
            else
                J2 = I + K + 1 - max( 1, K+I0-M )*KA1
            end
            # finish applying rotations in 2nd set from the left
            for L = KB-K:-1:1
                NRT = ( J2+KA+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J1T+L-2)), INCA, pointer(AB, KA1-L+LDAB*(J1T+L-2)), INCA, pointer(WORK, N+M-KB+J1T+KA), pointer(WORK, M-KB+J1T+KA), KA1)
                end
            end
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            for J = J1:KA1:J2
                WORK[M-KB+J] = WORK[M-KB+J+KA]
                WORK[N+M-KB+J] = WORK[N+M-KB+J+KA]
            end
            for J = J1:KA1:J2
                # create nonzero element a(j+ka,j-1) outside the band
                # and store it in WORK(m-kb+j)
                WORK[M-KB+J] = WORK[M-KB+J]*AB[KA1, J-1]
                AB[KA1, J-1] = WORK[N+M-KB+J]*AB[KA1, J-1]
            end
            if UPDATE
                if I+K > KA1 && K ≤ KBT
                    WORK[M-KB+I+K-KA] = WORK[M-KB+I+K]
                end
            end
        end

        for K = KB:-1:1
            J2 = I + K + 1 - max( 1, K+I0-M )*KA1
            NR = ( J2+KA-1 ) ÷ KA1
            J1 = J2 - ( NR-1 )*KA1
            if NR > 0
                # generate rotations in 2nd set to annihilate elements
                # which have been created outside the band
                largv!(NR, pointer(AB, KA1+LDAB*(J1-1)), INCA, pointer(WORK, M-KB+J1), KA1, pointer(WORK, N+M-KB+J1), KA1)
                # apply rotations in 2nd set from the right
                for L = 1:KA-1
                    lartv!(NR, pointer(AB, L+1+LDAB*(J1-1)), INCA, pointer(AB, L+2+LDAB*(J1-2)), INCA, pointer(WORK, N+M-KB+J1), pointer(WORK, M-KB+J1), KA1)
                end
                # apply rotations in 2nd set from both sides to diagonal
                # blocks
                lar2v!(NR, pointer(AB, 1+LDAB*(J1-1)), pointer(AB, 1+LDAB*(J1-2)), pointer(AB, 2+LDAB*(J1-2)), INCA, pointer(WORK, N+M-KB+J1), pointer(WORK, M-KB+J1), KA1)
            end
            # start applying rotations in 2nd set from the left
            for L = KA-1:-1:KB-K+1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J1T-KA1+L-1)), INCA, pointer(AB, KA1-L+LDAB*(J1T-KA1+L-1)), INCA, pointer(WORK, N+M-KB+J1T), pointer(WORK, M-KB+J1T), KA1)
                end
            end

            if WANTX
                # post-multiply X by product of rotations in 2nd set
                for J = J1:KA1:J2
                    push!(X, Givens(J, J-1, WORK[N+M-KB+J], -WORK[M-KB+J]))
                end
            end
        end

        for K = 1:KB-1
            J2 = I + K + 1 - max( 1, K+I0-M+1 )*KA1
            # finish applying rotations in 1st set from the left
            for L = KB-K:-1:1
                NRT = ( J2+L-1 ) ÷ KA1
                J1T = J2 - ( NRT-1 )*KA1
                if NRT > 0
                    lartv!(NRT, pointer(AB, KA1-L+1+LDAB*(J1T-KA1+L-1)), INCA, pointer(AB, KA1-L+LDAB*(J1T-KA1+L-1)), INCA, pointer(WORK, N+J1T), pointer(WORK, J1T), KA1)
                end
            end
        end

        if KB > 1
            for J = 2:min(I+KB,M)-2*KA-1
                WORK[N+J] = WORK[N+J+KA]
                WORK[J] = WORK[J+KA]
            end
        end
    end
    @goto L490
end
