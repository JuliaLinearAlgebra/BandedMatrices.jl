"Check that vect is correctly specified"
function chkvect(vect::AbstractChar)
    if !(vect == 'N' || vect == 'U' || vect == 'V')
        throw(ArgumentError("vect argument must be 'N' (X is not returned), or 'U' or 'V' (X is returned)"))
    end
    vect
end

for (fname, elty) in ((:dlargv_,:Float64),
                      (:slargv_,:Float32),
                      (:zlargv_,:ComplexF64),
                      (:clargv_,:ComplexF32))
    @eval begin
        #       SUBROUTINE DLARGV( N, X, INCX, Y, INCY, C, INCC )
        #       .. Scalar Arguments ..
        #       INTEGER INCC, INCX, INCY, N
        #       .. Array Arguments ..
        #       DOUBLE PRECISION C( * ), X( * ), Y( * )
        function largv!(n::Int, x::Ptr{$elty}, incx::Int, y::Ptr{$elty}, incy::Int, c::Ptr{$elty}, incc::Int)
            ccall((@blasfunc($fname), liblapack), Nothing, (Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}), n, x, incx, y, incy, c, incc)
        end
    end
end

for (fname, elty) in ((:dlartv_,:Float64),
                      (:slartv_,:Float32),
                      (:zlartv_,:ComplexF64),
                      (:clartv_,:ComplexF32))
    @eval begin
        #       SUBROUTINE DLARTV( N, X, INCX, Y, INCY, C, S, INCC )
        #       .. Scalar Arguments ..
        #       INTEGER INCC, INCX, INCY, N
        #       .. Array Arguments ..
        #       DOUBLE PRECISION C( * ), S( * ), X( * ), Y( * )
        function lartv!(n::Int, x::Ptr{$elty}, incx::Int, y::Ptr{$elty}, incy::Int, c::Ptr{$elty}, s::Ptr{$elty}, incc::Int)
            ccall((@blasfunc($fname), liblapack), Nothing, (Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}), n, x, incx, y, incy, c, s, incc)
        end
    end
end

for (fname, elty) in ((:drot_,:Float64),
                      (:srot_,:Float32),
                      (:zrot_,:ComplexF64),
                      (:crot_,:ComplexF32))
    @eval begin
        #       SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
        #       .. Scalar Arguments ..
        #       DOUBLE PRECISION C,S
        #       INTEGER INCX,INCY,N
        #       .. Array Arguments ..
        #       DOUBLE PRECISION DX(*),DY(*)
        function rot!(n::Int, x::Ptr{$elty}, incx::Int, y::Ptr{$elty}, incy::Int, c::$elty, s::$elty)
            ccall((@blasfunc($fname), libblas), Nothing, (Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty}), n, x, incx, y, incy, c, s)
        end
    end
end

for (fname, elty) in ((:dlartg_,:Float64),
                      (:slartg_,:Float32),
                      (:zlartg_,:ComplexF64),
                      (:clartg_,:ComplexF32))
    @eval begin
        #       SUBROUTINE DLARTG( F, G, CS, SN, R )
        #       .. Scalar Arguments ..
        #       DOUBLE PRECISION CS, F, G, R, SN
        function lartg!(f::$elty, g::$elty, cs::Ref{$elty}, sn::Ref{$elty}, r::Ref{$elty})
            ccall((@blasfunc($fname), liblapack), Nothing, (Ref{$elty}, Ref{$elty}, Ref{$elty}, Ref{$elty}, Ref{$elty}), f, g, cs, sn, r)
        end
    end
end

for (fname, elty) in ((:dlar2v_,:Float64),
                      (:slar2v_,:Float32),
                      (:zlar2v_,:ComplexF64),
                      (:clar2v_,:ComplexF32))
    @eval begin
        #       SUBROUTINE DLAR2V( N, X, Y, Z, INCX, C, S, INCC )
        #       .. Scalar Arguments ..
        #       INTEGER INCC, INCX, N
        #       .. Array Arguments ..
        #       DOUBLE PRECISION C( * ), S( * ), X( * ), Y( * ), Z( * )
        function lar2v!(n::Int, x::Ptr{$elty}, y::Ptr{$elty}, z::Ptr{$elty}, incx::Int, c::Ptr{$elty}, s::Ptr{$elty}, incc::Int)
            ccall((@blasfunc($fname), liblapack), Nothing, (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}), n, x, y, z, incx, c, s, incc)
        end
    end
end

## Symmetric tridiagonalization
for (fname, elty) in ((:dsbtrd_,:Float64),
                      (:ssbtrd_,:Float32))
    @eval begin
        function sbtrd!(vect::Char, uplo::Char,
                        m::Int, k::Int, A::AbstractMatrix{$elty},
                        d::AbstractVector{$elty}, e::AbstractVector{$elty}, Q::AbstractMatrix{$elty},
                        work::AbstractVector{$elty})
            require_one_based_indexing(A)
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
            require_one_based_indexing(A)
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
            A, info[]
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
            require_one_based_indexing(A)
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
            require_one_based_indexing(A)
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
            require_one_based_indexing(AB, BB, X, work)
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
