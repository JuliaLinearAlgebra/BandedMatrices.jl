normalize!(w) = BLAS.scal!(length(w),inv(norm(w)),w,1)
function normalize!(n,w)
    BLAS.scal!(n,inv(BLAS.nrm2(n,w,1)),w,1)
end


immutable BandedQR{T} <: Factorization{T}
    H::Matrix{T}  # Represents the orthogonal matrix Q
    R::BandedMatrix{T}
end

Base.size(QR::BandedQR,k) = size(QR.R,k)

immutable BandedQ{T} <: AbstractMatrix{T}
    H::Matrix{T}
    m::Int   # number of rows/cols
end


size(A::BandedQ) = (A.m, A.m)
size(Q::BandedQ,i::Integer) = i <= 0 ? error("dimension out of range") :
                                i == 1 ? A.m :
                                i == 2 ? A.m : 1


Base.At_mul_B(A::BandedQ,B::Vector) = At_mul_B!(similar(B),A,B)

function Base.At_mul_B!{T<:BlasFloat}(Y::Vector{T},A::BandedQ{T},B::Vector{T})
    if length(Y) != size(A,1) || length(B) != size(A,2)
        throw(DimensionMismatch("Matrices have wrong dimensions"))
    end

    H=A.H
    m=A.m

    M=size(H,1)
    b=pointer(B)
    y=pointer(Y)
    h=pointer(H)
    st=stride(H,2)

    sz=sizeof(T)

    BLAS.blascopy!(m,B,1,Y,1)


    for k=1:min(size(H,2),m-M+1)
        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=BLAS.dot(M,yp,1,wp,1)
        BLAS.axpy!(M,-2*dt,wp,1,yp,1)
    end

    for k=m-M+2:size(H,2)
        p=k-m+M-1

        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=BLAS.dot(M-p,yp,1,wp,1)
        BLAS.axpy!(M-p,-2*dt,wp,1,yp,1)
    end
    Y
end


function Base.getindex(QR::BandedQR,d::Symbol)
    d == :Q && return BandedQ(QR.H,size(QR,1))
    d == :R && return QR.R

    throw(KeyError(d))
end



function Base.qrfact(A::BandedMatrix)
    R=bzeros(eltype(A),size(A,1),size(A,2),A.l,A.l+A.u)
    R.data[A.l+1:end,:]=A.data
    banded_qrfact!(R)
end


function banded_qrfact!(R::BandedMatrix)
    T=eltype(R)
    M=R.l+1   # number of diag+subdiagonal bands
    m,n=size(R)
    W=Array(T,M,(n<m?n:m-1))
    w=pointer(W)
    r=pointer(R.data)
    sz=sizeof(T)
    st=stride(R.data,2)
    stw=stride(W,2)

    for k=1:min(size(R,1)-R.l,n)
        v=r+sz*(R.u + (k-1)*st)    # diagonal entry
        wp=w+stw*sz*(k-1)          # k-th column of W
        BLAS.blascopy!(M,v,1,wp,1)
        W[1,k]-= BLAS.nrm2(M,wp,1)
        normalize!(M,wp)

        for j=k:min(k+R.u,n)
            v=r+sz*(R.u + (k-1)*st + (j-k)*(st-1))
            dt=BLAS.dot(M,v,1,wp,1)
            BLAS.axpy!(M,-2*dt,wp,1,v,1)
        end
    end

    for k=m-R.l+1:(n<m?n:m-1)
        p=k-m+R.l
        v=r+sz*(R.u + (k-1)*st)    # diagonal entry
        wp=w+stw*sz*(k-1)          # k-th column of W
        BLAS.blascopy!(M-p,v,1,wp,1)
        W[1,k]-= BLAS.nrm2(M-p,wp,1)
        normalize!(M-p,wp)

        for j=k:min(k+R.u,n)
            v=r+sz*(R.u + (k-1)*st + (j-k)*(st-1))
            dt=BLAS.dot(M-p,v,1,wp,1)
            BLAS.axpy!(M-p,-2*dt,wp,1,v,1)
        end
    end

    BandedQR(W,R)
end
