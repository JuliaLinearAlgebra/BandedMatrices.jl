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
        function gbmv!(trans::Char, m::Int, kl::Int, ku::Int, alpha::($elty),
                       A::Ptr{$elty}, n::Int, st::Int,
                       x::Ptr{$elty}, incx::Int, beta::($elty), y::Ptr{$elty}, incy::Int)
            ccall((@blasfunc($fname), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                 Ptr{BlasInt}),
                 &trans, &m, &n, &kl,
                 &ku, &alpha, A, &st,
                 x, &incx, &beta, y, &incy)
            y
        end

        gbmv!(trans::Char, m::Int, kl::Int, ku::Int, alpha::($elty),
                       A::Ptr{$elty}, n::Int, st::Int,
                       x::Ptr{$elty}, beta::($elty), y::Ptr{$elty}) =
            gmv!(trans, m, kl, ku, alpha, A, n, st, x, 1, beta, y, 1)
    end
end


# #TODO: Speed up the following
# function gbmv!{T}(trans::Char, m::Integer, kl::Integer, ku::Integer, alpha::T, A::Ptr{T}, n::Integer, st::Integer, x::Ptr{T}, beta::T, y::Ptr{T})
#     data=pointer_to_array(A,(kl+ku+1,n))
#     xx=pointer_to_array(x,n)
#     yy=pointer_to_array(y,m)
#
#     B=BandedMatrix(data,m,kl,ku)
#
#     for j = 1:size(B,2), k = colrange(B,j)
#         yy[k] = beta*yy[k] + alpha*B[k,j]*xx[j]
#     end
#
#     yy
# end


gbmv!{T<:BlasFloat}(trans::Char, m::Int, kl::Int, ku::Int, alpha::T,
               A::StridedMatrix{T}, x::StridedVector{T}, beta::T, y::StridedVector{T}) =
    BLAS.gbmv!(trans,m,kl,ku,alpha,A,x,beta,y)

gbmv!{T<:BlasFloat}(trans::Char,α::T,A::AbstractMatrix{T},x::StridedVector{T},β::T,y::StridedVector{T}) =
    gbmv!(trans,size(A,1),bandwidth(A,1),bandwidth(A,2),α,
          pointer(A),size(A,2),leadingdimension(A),
          pointer(x),stride(x,1),β,pointer(y),stride(y,1))


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
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                 Ptr{$elty}, Ptr{BlasInt}),
                 &uplo, &n, &k,
                 &alpha, A, &lda,
                 x, &incx, &beta,
                 y, &incy)
            y
        end
    end
end

sbmv!{T<:BlasFloat}(uplo::Char, k::Int, alpha::T,
                    A::StridedMatrix{T}, x::StridedVector{T}, beta::T, y::StridedVector{T}) =
  BLAS.sbmv!(uplo,k,alpha,A,x,beta,y)


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
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8},
                 Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}),
                 &uplo, &trans, &diag,
                 &n, &k, A, &lda,
                 x, &incx)
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
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8},
                 Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}),
                 &uplo, &trans, &diag,
                 &n, &k, A, &lda,
                 x, &incx)
            x
        end
    end
end
