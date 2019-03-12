"Check that vect is correctly specified"
function chkvect(vect::AbstractChar)
    if !(vect == 'N' || vect == 'U' || vect == 'V')
        throw(ArgumentError("vect argument must be 'N' (X is not returned), or 'U' or 'V' (X is returned)"))
    end
    vect
end

## Symmetric tridiagonalization
for (fname, elty) in ((:dsbtrd_,:Float64),
                      (:ssbtrd_,:Float32))
    @eval begin
        function sbtrd!(vect::Char, uplo::Char,
                        m::Int, k::Int, A::AbstractMatrix{$elty},
                        d::AbstractVector{$elty}, e::AbstractVector{$elty}, Q::AbstractMatrix{$elty},
                        work::AbstractVector{$elty})
            @assert !has_offset_axes(A)
            chkstride1(A)
            chkuplo(uplo)
            chkvect(vect)
            info = Ref{BlasInt}()
            n    = size(A,2)
            n ≠ m && throw(ArgumentError("Matrix must be square"))
            size(A,1) < k+1 && throw(ArgumentError("Not enough bands"))
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8},
                 Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 vect, uplo,
                 n, k, A, max(1,stride(A,2)),
                 d, e, Q, max(1,stride(Q,2)), work, info)
            LAPACK.chklapackerror(info[])
            d, e, Q
        end
    end
end

## Bidiagonalization
for (fname, elty) in ((:dgbbrd_,:Float64),
                      (:sgbbrd_,:Float32))
    @eval begin
        function gbbrd!(vect::Char, m::Int, n::Int, ncc::Int,
                        kl::Int, ku::Int, ab::AbstractMatrix{$elty},
                        d::AbstractVector{$elty}, e::AbstractVector{$elty}, Q::AbstractMatrix{$elty},
                        Pt::AbstractMatrix{$elty}, C::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 vect, m, n, ncc,
                 kl, ku, ab, max(1,stride(ab,2)),
                 d, e, Q, max(1,stride(Q,2)),
                 Pt, max(1,stride(Pt,2)), C, max(1,stride(C,2)), work, info)
            LAPACK.chklapackerror(info[])
            d, e, Q, Pt, C
        end
    end
end


# (GB) general banded matrices, LU decomposition
for (gbtrf, elty) in
    ((:dgbtrf_, :Float64),
     (:sgbtrf_, :Float32),
     (:zgbtrf_, :ComplexF64),
     (:cgbtrf_, :ComplexF32))
    @eval begin
        # SUBROUTINE DGBTRF( M, N, KL, KU, AB, LDAB, IPIV, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, KL, KU, LDAB, M, N
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * )
        function gbtrf!(m::Int, l::Int, u::Int, A::AbstractMatrix{$elty},
                        ipiv::AbstractVector{BlasInt})
            @assert !has_offset_axes(A)
            chkstride1(A)
            info = Ref{BlasInt}()
            n    = size(A,2)
            size(A,1) < l+u+1 && throw(ArgumentError("Not enough bands"))
            length(ipiv) < min(m,n) && throw(ArgumentError("Not enough pivots"))
            info = Ref{BlasInt}()
            ccall((@blasfunc($gbtrf), liblapack), Nothing,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, l, u,
                  A, max(1,stride(A,2)), ipiv, info)
            LAPACK.chklapackerror(info[])
            A, ipiv
        end
    end
end


#     gbtrf!(m, kl, ku, AB) -> (AB, ipiv)
# Compute the LU factorization of a banded matrix `AB`. `kl` is the first
# subdiagonal containing a nonzero band, `ku` is the last superdiagonal
# containing one, and `m` is the first dimension of the matrix `AB`. Returns
# the LU factorization in-place and `ipiv`, the vector of pivots used.
# AB should be a (2l+u) by n matrix.

function gbtrf!(m::Int, l::Int, u::Int, AB::AbstractMatrix)
    ipiv = similar(AB, BlasInt, min(m, size(AB,2)))
    gbtrf!(m, l, u, AB, ipiv)
end


# (GB) general banded matrices, LU solver

#     gbtrs!(trans, kl, ku, m, AB, ipiv, B)
# Solve the equation `AB * X = B`. `trans` determines the orientation of `AB`. It may
# be `N` (no transpose), `T` (transpose), or `C` (conjugate transpose). `kl` is the
# first subdiagonal containing a nonzero band, `ku` is the last superdiagonal
# containing one, and `m` is the first dimension of the matrix `AB`. `ipiv` is the vector
# of pivots returned from `gbtrf!`. Returns the vector or matrix `X`, overwriting `B` in-place.


for (gbtrs, elty) in
    ((:dgbtrs_,:Float64),
     (:sgbtrs_,:Float32),
     (:zgbtrs_,:ComplexF64),
     (:cgbtrs_,:ComplexF32))
    @eval begin

        # SUBROUTINE DGBTRS( TRANS, N, KL, KU, NRHS, AB, LDAB, IPIV, B, LDB, INFO)
        # *     .. Scalar Arguments ..
        #       CHARACTER          TRANS
        #       INTEGER            INFO, KL, KU, LDAB, LDB, N, NRHS
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )
        function gbtrs!(trans::Char, m::Int, l::Int, u::Int,
                        A::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt},
                        B::AbstractVecOrMat{$elty})
            @assert !has_offset_axes(A)
            chkstride1(A)
            chktrans(trans)
            info = Ref{BlasInt}()
            n    = size(A,2)
            if m != n || m != size(B,1)
                throw(DimensionMismatch("matrix A has dimensions $(size(A)), but right hand side matrix B has dimensions $(size(B))"))
            end
            size(A,1) < 2*l+u+1 && throw(ArgumentError("not enough bands"))
            info = Ref{BlasInt}()

            ccall((@blasfunc($gbtrs), liblapack), Nothing,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  trans, m, l, u, size(B,2),
                  A, max(1,stride(A,2)), ipiv,
                  B, max(1,stride(B,2)), info)
            LAPACK.chklapackerror(info[])
            B
        end
    end
end

# All the eigenvalues and, optionally, eigenvectors of a real symmetric band matrix A.
for (fname, elty) in ((:dsbev_,:Float64),
                      (:ssbev_,:Float32))
    @eval begin
                # SUBROUTINE       SUBROUTINE DSBEV( JOBZ, UPLO, N, KD, AB, LDAB, W, Z, LDZ, WORK,
                #     $                  INFO )
                # CHARACTER          JOBZ, UPLO
                # INTEGER            INFO, KD, LDAB, LDZ, N
                # DOUBLE PRECISION   AB( LDAB, * ), W( * ), WORK( * ), Z( LDZ, * )

        function sbev!(jobz::Char, uplo::Char, n::Int, kd::Int, AB::AbstractMatrix{$elty},
                       w::AbstractVector{$elty}, Z::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                 Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 jobz, uplo, n, kd, AB, max(stride(AB,2),1), w, Z, max(stride(Z,2),1), work, info)
            LAPACK.chklapackerror(info[])
            w, Z
        end
    end
end

# All the generalized eigenvalues and, optionally, eigenvectors of a real symmetric-definite pencil (A,B).
for (fname, elty) in ((:dsbgv_,:Float64),
                      (:ssbgv_,:Float32))
    @eval begin
                # SUBROUTINE       DSBGV( JOBZ, UPLO, N, KA, KB, AB, LDAB, BB, LDBB, W, Z,
                #     $                  LDZ, WORK, INFO )
                # CHARACTER          JOBZ, UPLO
                # INTEGER            INFO, KA, KB, LDAB, LDBB, LDZ, N
                # DOUBLE PRECISION   AB( LDAB, * ), BB( LDBB, * ), W( * ), WORK( * ), Z( LDZ, * )

        function sbgv!(jobz::Char, uplo::Char, n::Int, ka::Int, kb::Int, AB::AbstractMatrix{$elty},
                       BB::AbstractMatrix{$elty}, w::AbstractVector{$elty}, Z::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 jobz, uplo, n, ka, kb, AB, max(stride(AB,2),1), BB, max(stride(BB,2),1), w, Z, max(stride(Z,2),1), work, info)
            LAPACK.chklapackerror(info[])
            w, Z
        end
    end
end



# Symmetric/Hermitian Positive Definite banded Cholesky factorization
for (fname, elty) in ((:dpbtrf_,:Float64),
                      (:spbtrf_,:Float32),
                      (:zpbtrf_,:ComplexF64),
                      (:cpbtrf_,:ComplexF32))
    @eval begin
                # SUBROUTINE DPBTRF( UPLO, N, KD, AB, LDAB, INFO )
                # CHARACTER          UPLO
                # INTEGER            INFO, KD, LDAB, N
                # DOUBLE PRECISION   AB( LDAB, * )

        function pbtrf!(uplo::Char, m::Int, kd::Int, A::AbstractMatrix{$elty})
            @assert !has_offset_axes(A)
            chkstride1(A)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            n    = size(A,2)
            n ≠ m && throw(ArgumentError("Matrix must be square"))
            size(A,1) < kd+1 && throw(ArgumentError("Not enough bands"))
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                 uplo, n, kd,
                 A, max(1,stride(A,2)), info)
            LAPACK.chklapackerror(info[])
            A
        end
    end
end

# Symmetric/Hermitian Positive Definite banded Cholesky solver
for (fname, elty) in ((:dpbtrs_,:Float64),
                      (:spbtrs_,:Float32),
                      (:zpbtrs_,:ComplexF64),
                      (:cpbtrs_,:ComplexF32))
    @eval begin
                # SUBROUTINE DPBTRS( UPLO, N, KD, NRHS, AB, LDAB, B, LDB, INFO )
                # CHARACTER          UPLO
                # INTEGER            INFO, KD, LDAB, LDB, N
                # DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )

        function pbtrs!(uplo::Char, m::Int, kd::Int, A::AbstractMatrix{$elty},
                        B::AbstractVecOrMat{$elty})
            @assert !has_offset_axes(A)
            chkstride1(A)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            n    = size(A,2)
            if m != n || m != size(B,1)
                throw(DimensionMismatch("matrix A has dimensions $(size(A)), but right hand side matrix B has dimensions $(size(B))"))
            end
            size(A,1) < kd+1 && throw(ArgumentError("Not enough bands"))
            ccall((@blasfunc($fname), liblapack), Nothing,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}),
                  uplo, n, kd, size(B,2),
                  A, max(1,stride(A,2)),
                  B, max(1,stride(B,2)),
                  info)
            LAPACK.chklapackerror(info[])
            B
        end
    end
end

# Symmetric/Hermitian Positive Definite split-Cholesky factorization
for (fname, elty) in ((:dpbstf_,:Float64),
                      (:spbstf_,:Float32),
                      (:zpbstf_,:ComplexF64),
                      (:cpbstf_,:ComplexF32))
    @eval begin
                # SUBROUTINE DPBSTF( UPLO, N, KD, AB, LDAB, INFO )
                # CHARACTER          UPLO
                # INTEGER            INFO, KD, LDAB, N
                # DOUBLE PRECISION   AB( LDAB, * )

        function pbstf!(uplo::Char, m::Int, kd::Int, A::AbstractMatrix{$elty})
            @assert !has_offset_axes(A)
            chkstride1(A)
            chkuplo(uplo)
            info = Ref{BlasInt}()
            n    = size(A,2)
            n ≠ m && throw(ArgumentError("Matrix must be square"))
            size(A,1) < kd+1 && throw(ArgumentError("Not enough bands"))
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                 uplo, n, kd,
                 A, max(1,stride(A,2)), info)
            LAPACK.chklapackerror(info[])
            A
        end
    end
end

# Convert Symmetric Positive Definite generalized eigenvalue problem
# to a symmetric eigenvalue problem assuming B has been processed by
# a split-Cholesky factorization.
for (fname, elty) in ((:dsbgst_,:Float64),
                      (:ssbgst_,:Float32))
    @eval begin
                # SUBROUTINE DSBGST( VECT, UPLO, N, KA, KB, AB, LDAB,
                # BB, LDBB, X, LDX, WORK, INFO )
                # CHARACTER          UPLO, VECT
                # INTEGER            INFO, KA, KB, LDAB, LDBB, LDX, N
                # DOUBLE PRECISION   AB( LDAB, * ), BB( LDBB, * ), WORK( * ),
                # X( LDX, * )

        function sbgst!(vect::Char, uplo::Char, n::Int, ka::Int, kb::Int,
                         AB::AbstractMatrix{$elty}, BB::AbstractMatrix{$elty},
                         X::AbstractVecOrMat{$elty}, work::AbstractVector{$elty})
            @assert !has_offset_axes(AB, BB, X, work)
            chkstride1(AB, BB)
            chkuplo(uplo)
            chkvect(vect)
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                 Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}),
                 vect, uplo, n, ka,
                 kb, AB, max(stride(AB,2),1), BB,
                 max(stride(BB,2),1), X, max(1,stride(X,2)), work,
                 info)
            LAPACK.chklapackerror(info[])
            AB
        end
    end
end
