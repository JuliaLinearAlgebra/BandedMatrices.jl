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
size(A::BandedQ,i::Integer) = i <= 0 ? error("dimension out of range") :
                                i == 1 ? A.m :
                                i == 2 ? A.m : 1


Base.At_mul_B{T<:Real}(A::BandedQ{T},B::Union{Vector{T},Matrix{T}}) = Ac_mul_B(A,B)
Base.At_mul_B!{T<:Real}(Y,A::BandedQ{T},B::Union{Vector{T},Matrix{T}}) = Ac_mul_B!(Y,A,B)

Base.Ac_mul_B{T<:Real,V<:Complex}(A::BandedQ{T},B::Vector{V}) =
    Ac_mul_B(A,real(B))+im*Ac_mul_B(A,imag(B))
Base.Ac_mul_B{T<:Real,V<:Real}(A::BandedQ{T},B::Vector{V}) =
    Ac_mul_B(A,Vector{T}(B))
Base.Ac_mul_B{T<:Complex}(A::BandedQ{T},B::Vector) =
    Ac_mul_B(A,Vector{T}(B))
Base.Ac_mul_B{T<:Real}(A::BandedQ{T},B::Vector{T}) = Ac_mul_B!(similar(B),A,B)
Base.Ac_mul_B{T<:Complex}(A::BandedQ{T},B::Vector{T}) = Ac_mul_B!(similar(B),A,B)

function Base.Ac_mul_B!{T<:BlasFloat}(Y::Vector{T},A::BandedQ{T},B::Vector{T})
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

        dt=dot(M,wp,1,yp,1)
        BLAS.axpy!(M,-2*dt,wp,1,yp,1)
    end

    for k=m-M+2:size(H,2)
        p=k-m+M-1

        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=dot(M-p,wp,1,yp,1)
        BLAS.axpy!(M-p,-2*dt,wp,1,yp,1)
    end
    Y
end


# Each householder is symmetyric, this just reverses the order of application
function Base.A_mul_B!{T<:BlasFloat}(Y::Vector{T},A::BandedQ{T},B::Vector{T})
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

    for k=size(H,2):-1:m-M+2
        p=k-m+M-1

        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=dot(M-p,wp,1,yp,1)
        BLAS.axpy!(M-p,-2*dt,wp,1,yp,1)
    end


    for k=min(size(H,2),m-M+1):-1:1
        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=dot(M,wp,1,yp,1)
        BLAS.axpy!(M,-2*dt,wp,1,yp,1)
    end

    Y
end

function Base.A_mul_B!(Y::Matrix,A::BandedQ,B::Matrix)
    for j=1:size(A,2)
        Y[:,j]=A*B[:,j]
    end
    Y
end

function Base.Ac_mul_B!(Y::Matrix,A::BandedQ,B::Matrix)
    for j=1:size(A,2)
        Y[:,j]=A'*B[:,j]
    end
    Y
end

Base.full(A::BandedQ) = A*eye(eltype(A),size(A,1))

Base.linearindexing{T}(::Type{BandedQ{T}}) = Base.LinearSlow()
Base.getindex(A::BandedQ,k::Int,j::Int) = (A*eltype(A)[zeros(j-1);1.0;zeros(size(A,2)-j)])[k]




function Base.getindex(QR::BandedQR,d::Symbol)
    d == :Q && return BandedQ(QR.H,size(QR,1))
    d == :R && return QR.R

    throw(KeyError(d))
end



function Base.qr(A::BandedMatrix)
    QR = qrfact(A)
    QR[:Q],QR[:R]
end

function Base.qrfact(A::BandedMatrix)
    R=bzeros(eltype(A),size(A,1),size(A,2),A.l,A.l+A.u)
    R.data[A.l+1:end,:]=A.data
    banded_qrfact!(R)
end

flipsign(x,y) = Base.flipsign(x,y)
flipsign(x,y::Complex) = y==0?x:x*sign(y)

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
        W[1,k]+= flipsign(BLAS.nrm2(M,wp,1),W[1,k])
        normalize!(M,wp)

        for j=k:min(k+R.u,n)
            v=r+sz*(R.u + (k-1)*st + (j-k)*(st-1))
            dt=dot(M,wp,1,v,1)
            BLAS.axpy!(M,-2*dt,wp,1,v,1)
        end
    end

    for k=m-R.l+1:(n<m?n:m-1)
        p=k-m+R.l
        v=r+sz*(R.u + (k-1)*st)    # diagonal entry
        wp=w+stw*sz*(k-1)          # k-th column of W
        BLAS.blascopy!(M-p,v,1,wp,1)
        W[1,k]+= flipsign(BLAS.nrm2(M-p,wp,1),W[1,k])
        normalize!(M-p,wp)

        for j=k:min(k+R.u,n)
            v=r+sz*(R.u + (k-1)*st + (j-k)*(st-1))
            dt=dot(M-p,wp,1,v,1)
            BLAS.axpy!(M-p,-2*dt,wp,1,v,1)
        end
    end

    BandedQR(W,BandedMatrix(R.data[1:R.u+1,:],m,0,R.u))
end
