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
function gbmm!(alpha,A::BandedMatrix,B::Matrix,beta,C)
    st=max(1,stride(A.data,2))
    n=size(A.data,2)
    p=pointer(A.data)


    x=pointer(B)
    sx=sizeof(eltype(B))
    ν,m=size(B)

    @assert size(C,1)==n
    @assert size(C,2)==m

    y=pointer(C)
    sy=sizeof(eltype(C))

    for k=1:m
        gbmv!('T',A.m,A.u,A.l,alpha,p,n,st,x+(k-1)*sx*ν,beta,y+(k-1)*sy*n)
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

Base.A_mul_B!(Y::Matrix,A::BandedMatrix,B::Matrix)=gbmm!(1.0,A,B,0.,Y)

## Matrix*Vector Multiplicaiton

function Base.A_mul_B!(c::Vector,A::BandedMatrix,b::Vector)
    for k=1:size(A,1)  # rows of c
        @simd for l=max(1,k-A.l):min(k+A.u,size(A,2)) # columns of A/rows of b
             @inbounds c[k]+=A.data[l-k+A.l+1,k]*b[l]
        end
    end
    c
end





## Matrix*Matrix Multiplication




function bmultiply!(C::BandedMatrix,A::BandedMatrix,B::BandedMatrix,ri::Integer=0,ci::Integer=0,rs::Integer=1,cs::Integer=1)
    n=size(A,1);m=size(B,2)
    @assert size(C,1)≥rs*n+ri&&size(C,2)≥cs*m+ci
    for k=1:n  # rows of C
        for l=max(1,k-A.l):min(k+A.u,size(A,2)) # columns of A
            @inbounds Aj=A.data[l-k+A.l+1,k]


            #  A[k,j] == A.data[j-k+A.l+1,k]
            shB=-l+B.l+1
            ks=rs*k+ri
            shC=ci-ks+C.l+1
            @simd for j=max(1,l-B.l):min(B.u+l,m) # columns of C/B
                @inbounds C.data[cs*j+shC,ks]+=Aj*B.data[j+shB,l]
            end
        end
    end
    C
end

function bmultiply!(C::Matrix,A::BandedMatrix,B::Matrix,ri::Integer=0,ci::Integer=0,rs::Integer=1,cs::Integer=1)
    n=size(A,1);m=size(B,2)
    @assert size(C,1)≥rs*n+ri&&size(C,2)≥cs*m+ci
    for k=1:n  # rows of C
        for l=max(1,k-A.l):min(k+A.u,size(A,2)) # columns of A
            @inbounds Aj=A.data[l-k+A.l+1,k]

             @simd for j=1:m # columns of C/B
                 @inbounds C[rs*k+ri,cs*j+ci]+=Aj*B[l,j]
             end
        end
    end
    C
end


Base.A_mul_B!(C::BandedMatrix,A::BandedMatrix,B::BandedMatrix)=bmultiply!(C,A,B)
