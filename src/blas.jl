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
function gbmm!{T}(alpha,A::BandedMatrix{T},B::BandedMatrix{T},beta,C::BandedMatrix{T})
    @assert size(A,1)==size(C,1)
    @assert size(A,2)==size(B,1)
    @assert size(C,2)==size(B,2)


    a=pointer(A.data)
    b=pointer(B.data)
    c=pointer(C.data)
    n,m=size(A)
    sta=max(1,stride(A.data,2))
    stb=max(1,stride(B.data,2))
    stc=max(1,stride(C.data,2))
    sz=sizeof(T)
    #Multiply columns where index of B starts from 1
    for j=1:(B.u+1)
        gbmv!('N',C.l+j, A.l, A.u, alpha, a, B.l+j, sta, b+sz*((j-1)*stb+B.u-j+1), beta, c+sz*((j-1)*stc+C.u-j+1))
    end
    #Multiply columns where index of C starts from 1
    for j=B.u+2:C.u+1
        p=j-B.u-1
        gbmv!('N', C.l+j, A.l+p, A.u-p, alpha, a+sz*p*sta, B.l+B.u+1, sta, b+sz*(j-1)*stb, beta, c+sz*((j-1)*stc+C.u-j+1))
    end

    # multiply columns where A, B and C are mid
    for j=C.u+2:n-C.l
        gbmv!('N', C.l+C.u+1, A.l+A.u, 0, alpha, a+sz*(j-B.u-1)*sta, B.l+B.u+1, sta, b+sz*(j-1)*stb, beta, c+sz*(j-1)*stc)
    end

    # multiply columns where A and B are mid and C is bottom
    for j=n-C.l+1:m-B.l
        p=j-n+C.l
        gbmv!('N', C.l+C.u+1-p, A.l+A.u, 0, alpha, a+sz*(j-B.u-1)*sta, B.l+B.u+1, sta, b+sz*(j-1)*stb, beta, c+sz*(j-1)*stc)
    end

    # multiply columns where A,  B and C are bottom
    for j=m-B.l+1:size(B,2)
        p=j-n+C.l
        gbmv!('N', C.l+C.u+1-p, A.l+A.u, 0, alpha, a+sz*(j-B.u-1)*sta, B.l+B.u+1-(j-m+B.l), sta, b+sz*(j-1)*stb, beta, c+sz*(j-1)*stc)
    end

    C
end


function gbmm!{T}(alpha,A::BandedMatrix{T},B::Matrix{T},beta,C::Matrix{T})
    st=max(1,stride(A.data,2))
    n=size(A.data,2)
    p=pointer(A.data)


    x=pointer(B)
    sx=sizeof(eltype(B))
    ν,m=size(B)

    @assert size(C,1)==n
    @assert size(C,2)==m

    c=pointer(C)
    sc=sizeof(eltype(C))

    for k=1:m
        gbmv!('N',A.m,A.l,A.u,alpha,p,n,st,x+(k-1)*sx*ν,beta,c+(k-1)*sc*n)
    end
    C
end

function Base.BLAS.axpy!(a::Number,X::BandedMatrix,Y::BandedMatrix)
    @assert size(X)==size(Y)
    @assert X.l ≤ Y.l && X.u ≤ Y.u
    for (k,j) in eachbandedindex(X)
        unsafe_pluseq!(Y,a*unsafe_getindex(X,k,j),k,j)
    end
    Y
end


## A_mul_B! overrides

Base.A_mul_B!(C::Matrix,A::BandedMatrix,B::Matrix)=gbmm!(1.0,A,B,0.,C)

## Matrix*Vector Multiplicaiton



Base.A_mul_B!(c::Vector,A::BandedMatrix,b::Vector)=BLAS.gbmv!('N',A.m,A.l,A.u,1.0,A.data,b,0.,c)



## Matrix*Matrix Multiplication


Base.A_mul_B!(C::BandedMatrix,A::BandedMatrix,B::BandedMatrix)=gbmm!(1.0,A,B,0.0,C)
