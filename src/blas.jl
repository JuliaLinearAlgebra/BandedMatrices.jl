# We implement for pointers

import Base.BLAS: blasfunc,libblas,BlasInt

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
        function gbmv!(trans::Char, m::Integer, kl::Integer, ku::Integer, alpha::($elty), A::Ptr{$elty}, n::Integer, st::Integer, x::Ptr{$elty}, beta::($elty), y::Ptr{$elty})
            ccall(($(blasfunc(fname)), libblas), Void,
                (Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                 Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                 Ptr{BlasInt}),
                 &trans, &m, &n, &kl,
                 &ku, &alpha, A, &st,
                 x, &1, &beta, y, &1)
            y
        end
    end
end



# this is matrix*matrix
function mm!(alpha,A::BandedMatrix,B::Matrix,beta,C)

    st=max(1,stride(A.data,2))
    n=size(A.data,2)
    p=pointer(A.data)

    x=pointer(B)
    sx=sizeof(eltype(B))
    m=size(B,2)

    y=pointer(C)
    sy=sizeof(eltype(C))

    for k=1:m
        BLAS.gbmv!('T',A.m,A.u,A.l,alpha,p,n,st,x+(k-1)*sx*m,beta,y+(k-1)*sy*n)
    end
    C
end


