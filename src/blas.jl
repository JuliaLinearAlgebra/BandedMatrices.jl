import LinearAlgebra.BLAS.@blasfunc


for (fname, elty) in ((:dgbmv_,:Float64),
                      (:sgbmv_,:Float32),
                      (:zgbmv_,:ComplexF64),
                      (:cgbmv_,:ComplexF32))
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
            ccall((@blasfunc($fname), libblas), Nothing,
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




for (fname, elty) in ((:dsbmv_,:Float64),
                      (:ssbmv_,:Float32),
                      (:zsbmv_,:ComplexF64),
                      (:csbmv_,:ComplexF32))
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
            ccall((@blasfunc($fname), libblas), Nothing,
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
      A::AbstractMatrix{T}, x::AbstractVector{T}, beta::T, y::AbstractVector{T}) where {T<:BlasFloat} =
    sbmv!(uplo,size(A,2),k,alpha,pointer(A),max(1,stride(A,2)),x,stride(x,1),beta,y,stride(y,1))


## Triangular *

function tbmv! end

for (fname, elty) in ((:dtbmv_,:Float64),
                        (:stbmv_,:Float32),
                        (:ztbmv_,:ComplexF64),
                        (:ctbmv_,:ComplexF32))
    @eval begin
                #       SUBROUTINE DTRMV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
                # *     .. Scalar Arguments ..
                #       INTEGER INCX,LDA,N
                #       CHARACTER DIAG,TRANS,UPLO
                # *     .. Array Arguments ..
                #       DOUBLE PRECISION A(LDA,*),X(*)
        function tbmv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                       m::Int, k::Int, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            require_one_based_indexing(A, x)
            n = size(A,2)
            size(A,1) ≥ k+1 || throw(ArgumentError("triangular banded data missing"))
            n == m || throw(DimensionMismatch("matrix is not square: dimensions are $n, $m"))
            if n != length(x)
                throw(DimensionMismatch("size of A is $n != length(x) = $(length(x))"))
            end
            chkstride1(A)
            ccall((@blasfunc($fname), libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, diag, m, k,
                 A, max(1,stride(A,2)), x, max(1,stride(x, 1)))
            x
        end
        function tbmv(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, m::Int, k::Int,
                      A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            tbmv!(uplo, trans, diag, m, k, A, copy(x))
        end
    end
end

## Triangular \

function tbsv end

for (fname, elty) in ((:dtbsv_,:Float64),
                        (:stbsv_,:Float32),
                        (:ztbsv_,:ComplexF64),
                        (:ctbsv_,:ComplexF32))
    @eval begin
                #       SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
                #       .. Scalar Arguments ..
                #       INTEGER INCX,LDA,N
                #       CHARACTER DIAG,TRANS,UPLO
                #       .. Array Arguments ..
                #       DOUBLE PRECISION A(LDA,*),X(*)
        function tbsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                        m::Int, k::Int, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            require_one_based_indexing(A, x)
            n = size(A,2)
            size(A,1) ≥ k+1 || throw(ArgumentError("triangular banded data missing"))
            n == m || throw(DimensionMismatch("matrix is not square: dimensions are $n, $m"))
            if n != length(x)
                throw(DimensionMismatch("size of A is $n != length(x) = $(length(x))"))
            end
            chkstride1(A)
            ccall((@blasfunc($fname), libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, diag, m, k,
                 A, max(1,stride(A,2)), x, stride(x, 1))
            x
        end
        function tbsv(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar,
                        m::Int, k::Int, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            tbsv!(uplo, trans, diag, m, k, A, copy(x))
        end
    end
end
