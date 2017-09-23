import Base.BLAS.@blasfunc

for (fname, elty) in ((:dgbmv_,:Float64),
                      (:sgbmv_,:Float32),
                      (:zgbmv_,:Complex128),
                      (:cgbmv_,:Complex64))
    @eval begin
             # SUBROUTINE DGBMV(TRANS,M,N,KL,KU,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
             # *     .. Scalar Arguments ..
             #       DOUBLE PRECISION ALPHA,BETA
             #       INTEGER INCX,INCY,KL,KU,LDA,M,N
             #       CHARACTER TRANS
             # *     .. Array Arguments ..
             #       DOUBLE PRECISION A(LDA,*),X(*),Y(*)
        function gbmv!(trans::Char, m::Int, n::Int, kl::Int, ku::Int, alpha::($elty),
                       A::Ptr{$elty}, st::Int,
                       x::Ptr{$elty}, incx::Int, beta::($elty), y::Ptr{$elty}, incy::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                 Ref{BlasInt}),
                 trans, m, n, kl,
                 ku, alpha, A, st,
                 x, incx, beta, y, incy)
            y
        end
    end
end

gbmv!(trans::Char, m::Int, kl::Int, ku::Int, alpha::T,
               A::StridedMatrix{T}, x::StridedVector{T}, beta::T, y::StridedVector{T}) where {T<:BlasFloat} =
    gbmv!(trans,m,size(A,2),kl,ku,alpha,
          pointer(A),max(1,stride(A,2)),pointer(x),stride(x,1),beta,y,stride(y,1))


for (fname, elty) in ((:dsbmv_,:Float64),
                      (:ssbmv_,:Float32),
                      (:zsbmv_,:Complex128),
                      (:csbmv_,:Complex64))
    @eval begin
                # SUBROUTINE DSBMV(UPLO, N, K, ALPHA, A, LDA,
                # X, INCX, BETA, Y, INCY)
                # DOUBLE PRECISION ALPHA,BETA
                # INTEGER INCX,INCY,K,LDA,N
                # CHARACTER*1 UPLO
                # DOUBLE PRECISION A(LDA,*), X(*), Y(*)

        function sbmv!(uplo::Char, n::Int, k::Int, alpha::($elty),
                       A::Ptr{$elty}, lda::Int,
                       x::Ptr{$elty}, incx::Int, beta::($elty), y::Ptr{$elty}, incy::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ref{$elty},
                 Ptr{$elty}, Ref{BlasInt}),
                 uplo, n, k,
                 alpha, A, lda,
                 x, incx, beta,
                 y, incy)
            y
        end
    end
end

sbmv!(uplo::Char, k::Int, alpha::T,
      A::StridedMatrix{T}, x::StridedVector{T}, beta::T, y::StridedVector{T}) where {T<:BlasFloat} =
    sbmv!(uplo,size(A,2),k,alpha,pointer(A),max(1,stride(A,2)),x,stride(x,1),beta,y,stride(y,1))


## Triangular *

for (fname, elty) in ((:dtbmv_,:Float64),
                      (:stbmv_,:Float32),
                      (:ztbmv_,:Complex128),
                      (:ctbmv_,:Complex64))
    @eval begin
        function tbmv!(uplo::Char, trans::Char, diag::Char,
                        n::Int, k::Int, A::Ptr{$elty}, lda::Int,
                        x::Ptr{$elty}, incx::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                 Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, diag,
                 n, k,
                 A, lda,
                 x, incx)
            x
        end
    end
end

## Triangular \

for (fname, elty) in ((:dtbsv_,:Float64),
                      (:stbsv_,:Float32),
                      (:ztbsv_,:Complex128),
                      (:ctbsv_,:Complex64))
    @eval begin
        function tbsv!(uplo::Char, trans::Char, diag::Char,
                        n::Int, k::Int, A::Ptr{$elty}, lda::Int,
                        x::Ptr{$elty}, incx::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                 Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, diag,
                 n, k,
                 A, lda,
                 x, incx)
            x
        end
    end
end
