## Symmetric tridiagonalization
for (fname, elty) in ((:dsbtrd_,:Float64),
                      (:ssbtrd_,:Float32))
    @eval begin
        function sbtrd!(vect::Char, uplo::Char,
                        n::Int, k::Int, ab::Ptr{$elty}, ldab::Int,
                        d::Ptr{$elty}, e::Ptr{$elty}, q::Ptr{$elty}, ldq::Int,
                        work::Ptr{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8},
                 Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 &vect, &uplo,
                 &n, &k, ab, &ldab,
                 d, e, q, &ldq, work, info)
            d, e, q
        end
    end
end

# (GB) general banded matrices, LU decomposition
for (gbtrf, elty) in
    ((:dgbtrf_, :Float64),
     (:sgbtrf_, :Float32),
     (:zgbtrf_, :Complex128),
     (:cgbtrf_, :Complex64))
    @eval begin
        # SUBROUTINE DGBTRF( M, N, KL, KU, AB, LDAB, IPIV, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, KL, KU, LDAB, M, N
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * )
        function gbtrf!(m::Int, n::Int, kl::Int, ku::Int, AB::Ptr{$elty}, ldab::Int,
                        ipiv::Ptr{BlasInt})
            info = Ref{BlasInt}()
            ccall((@blasfunc($gbtrf), liblapack), Void,
                  (Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  &m, &n, &kl, &ku, AB, &ldab, ipiv, info)
            AB, ipiv
        end
    end
end


#     gbtrf!(kl, ku, m, AB) -> (AB, ipiv)
# Compute the LU factorization of a banded matrix `AB`. `kl` is the first
# subdiagonal containing a nonzero band, `ku` is the last superdiagonal
# containing one, and `m` is the first dimension of the matrix `AB`. Returns
# the LU factorization in-place and `ipiv`, the vector of pivots used.
# AB should be a (2l+u) by n matrix.

function gbtrf!(kl::Int, ku::Int, m::Int, AB::StridedMatrix)
    n = size(AB, 2)
    mnmn = min(m, n)
    ipiv = similar(AB, BlasInt, mnmn)
    gbtrf!(m, n, kl, ku, pointer(AB), max(1,stride(AB, 2)), pointer(ipiv))
    AB, ipiv
end


# (GB) general banded matrices, LU solver
for (gbtrs, elty) in
    ((:dgbtrs_,:Float64),
     (:sgbtrs_,:Float32),
     (:zgbtrs_,:Complex128),
     (:cgbtrs_,:Complex64))
    @eval begin

        # SUBROUTINE DGBTRS( TRANS, N, KL, KU, NRHS, AB, LDAB, IPIV, B, LDB, INFO)
        # *     .. Scalar Arguments ..
        #       CHARACTER          TRANS
        #       INTEGER            INFO, KL, KU, LDAB, LDB, N, NRHS
        # *     .. Array Arguments ..
        #       INTEGER            IPIV( * )
        #       DOUBLE PRECISION   AB( LDAB, * ), B( LDB, * )
        function gbtrs!(trans::Char, n::Int, kl::Int, ku::Int, nrhs::Int, m::Int,
                        AB::Ptr{$elty}, ldab::Int, ipiv::Ptr{BlasInt},
                        B::Ptr{$elty}, ldb::Int)
            info = Ref{BlasInt}()

            ccall((@blasfunc($gbtrs), liblapack), Void,
                  (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},   Ptr{BlasInt},
                   Ptr{BlasInt}),
                  &trans, &n, &kl, &ku, &nrhs, AB, &ldab, ipiv,
                  B, &ldb, info)
            B
        end
    end
end


#     gbtrs!(trans, kl, ku, m, AB, ipiv, B)
# Solve the equation `AB * X = B`. `trans` determines the orientation of `AB`. It may
# be `N` (no transpose), `T` (transpose), or `C` (conjugate transpose). `kl` is the
# first subdiagonal containing a nonzero band, `ku` is the last superdiagonal
# containing one, and `m` is the first dimension of the matrix `AB`. `ipiv` is the vector
# of pivots returned from `gbtrf!`. Returns the vector or matrix `X`, overwriting `B` in-place.

function gbtrs!(trans::Char, kl::Int, ku::Int, m::Int, AB::StridedMatrix,
                ipiv::StridedVector{BlasInt}, B::StridedVecOrMat)
    n    = size(AB,2)
    if m != n || m != size(B,1)
        throw(DimensionMismatch("matrix AB has dimensions $(size(AB)), but right hand side matrix B has dimensions $(size(B))"))
    end
    gbtrs!(trans, n, kl, ku, size(B,2), m, pointer(AB), max(1,stride(AB, 2)),
           pointer(ipiv), pointer(B), max(1,stride(B, 2)))
    B
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

        function sbev!(jobz::Char, uplo::Char, n::Int, kd::Int, AB::Ptr{$elty}, ldab::Int,
                       w::Ptr{$elty}, Z::Ptr{$elty}, ldz::Int, work::Ptr{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 &jobz, &uplo, &n, &kd, AB, &ldab, w, Z, &ldz, work, info)
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

        function sbgv!(jobz::Char, uplo::Char, n::Int, ka::Int, kb::Int, AB::Ptr{$elty}, ldab::Int,
                       BB::Ptr{$elty}, ldbb::Int, w::Ptr{$elty}, Z::Ptr{$elty}, ldz::Int, work::Ptr{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                 &jobz, &uplo, &n, &ka, &kb, AB, &ldab, BB, &ldbb, w, Z, &ldz, work, info)
            w, Z
        end
    end
end


# Symmetric/Hermitian Positive Definite banded Cholesky factorization
for (fname, elty) in ((:dpbtrf_,:Float64),
                      (:spbtrf_,:Float32),
                      (:zpbtrf_,:Complex128),
                      (:cpbtrf_,:Complex64))
    @eval begin
                # SUBROUTINE DPBTRF( UPLO, N, KD, AB, LDAB, INFO )
                # CHARACTER          UPLO
                # INTEGER            INFO, KD, LDAB, N
                # DOUBLE PRECISION   AB( LDAB, * )

        function pbtrf!(uplo::Char, n::Int, kd::Int, AB::Ptr{$elty}, ldab::Int)
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                 &uplo, &n, &kd, AB, &ldab, info)
            AB
        end
    end
end

# Symmetric/Hermitian Positive Definite split-Cholesky factorization
for (fname, elty) in ((:dpbstf_,:Float64),
                      (:spbstf_,:Float32),
                      (:zpbstf_,:Complex128),
                      (:cpbstf_,:Complex64))
    @eval begin
                # SUBROUTINE DPBSTF( UPLO, N, KD, AB, LDAB, INFO )
                # CHARACTER          UPLO
                # INTEGER            INFO, KD, LDAB, N
                # DOUBLE PRECISION   AB( LDAB, * )

        function pbstf!(uplo::Char, n::Int, kd::Int, AB::Ptr{$elty}, ldab::Int)
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                 &uplo, &n, &kd, AB, &ldab, info)
            AB
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
                         AB::Ptr{$elty}, ldab::Int, BB::Ptr{$elty}, ldbb::Int,
                         X::Ptr{$elty}, ldx::Int, work::Ptr{$elty})
            info  = Ref{BlasInt}()
            ccall((@blasfunc($fname), liblapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}),
                 &vect, &uplo, &n, &ka, &kb, AB, &ldab, BB, &ldbb,
                 X, &ldx, work, info)
            AB
        end
    end
end
