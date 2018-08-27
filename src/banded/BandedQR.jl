struct BandedQR{T} <: Factorization{T}
    H::Matrix{T}  # Represents the orthogonal matrix Q
    R::BandedMatrix{T}
end

size(QR::BandedQR,k) = size(QR.R,k)

struct BandedQ{T} <: AbstractMatrix{T}
    H::Matrix{T}
    m::Int   # number of rows/cols
end


size(A::BandedQ) = (A.m, A.m)
size(A::BandedQ,i::Integer) = i <= 0 ? error("dimension out of range") :
                                i == 1 ? A.m :
                                i == 2 ? A.m : 1

transpose(A::BandedQ) = Transpose(A)
adjoint(A::BandedQ) = Adjoint(A)

*(At::Transpose{T,BandedQ{T}}, B::AbstractVecOrMat{T}) where {T<:Real} = transpose(At)'*B
mul!(Y, At::Transpose{T,BandedQ{T}}, B::AbstractVecOrMat{T}) where {T<:Real} =
    mul!(Y, transpose(At)', B)

*(Ac::Adjoint{T,BandedQ{T}}, B::AbstractVector{V}) where {T<:Real,V<:Complex} =
    Ac * real(B) + im* Ac * imag(B)
*(Ac::Adjoint{T,BandedQ{T}}, B::AbstractVector{V}) where {T<:Real,V<:Real} =
    Ac * convert(Vector{T}, B)
*(Ac::Adjoint{T,BandedQ{T}}, B::AbstractVector) where {T<:Complex} =
    Ac * convert(Vector{T}, B)
*(Ac::Adjoint{T,BandedQ{T}}, B::AbstractVector{T}) where {T<:Real} =
    mul!(similar(B), Ac, B)
*(Ac::Adjoint{T,BandedQ{T}}, B::AbstractVector{T}) where {T<:Complex} =
    mul!(similar(B), Ac, B)

function mul!(Y::Vector{T}, Ac::Adjoint{T,BandedQ{T}}, B::Vector{T}) where {T<:BlasFloat}
    A = parent(Ac)
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
        axpy!(M,-2*dt,wp,1,yp,1)
    end

    for k=m-M+2:size(H,2)
        p=k-m+M-1

        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=dot(M-p,wp,1,yp,1)
        axpy!(M-p,-2*dt,wp,1,yp,1)
    end
    Y
end


# Each householder is symmetyric, this just reverses the order of application
function mul!(Y::Vector{T}, A::BandedQ{T}, B::Vector{T}) where {T<:BlasFloat}
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
        axpy!(M-p,-2*dt,wp,1,yp,1)
    end


    for k=min(size(H,2),m-M+1):-1:1
        wp=h+sz*st*(k-1)
        yp=y+sz*(k-1)

        dt=dot(M,wp,1,yp,1)
        axpy!(M,-2*dt,wp,1,yp,1)
    end

    Y
end

function mul!(Y::Matrix, A::BandedQ, B::Matrix)
    for j=1:size(A,2)
        Y[:,j]=A*B[:,j]
    end
    Y
end

function mul!(Y::Matrix,A::Adjoint{T,BandedQ{T}},B::Matrix) where T
    for j=1:size(A,2)
        Y[:,j]=A'*B[:,j]
    end
    Y
end

function (*)(A::BandedQ{T}, x::AbstractVector{S}) where {T,S}
    TS = promote_type(T, S)
    mul!(similar(x,TS,size(A,1)),A,x)
end

Base.IndexStyle(::Type{BandedQ{T}}) where {T} = IndexCartesian()
Base.getindex(A::BandedQ,k::Int,j::Int) = (A*eltype(A)[zeros(j-1);1.0;zeros(size(A,2)-j)])[k]




function Base.getindex(QR::BandedQR,d::Symbol)
    d == :Q && return BandedQ(QR.H,size(QR,1))
    d == :R && return QR.R

    throw(KeyError(d))
end



function getproperty(QR::BandedQR, d::Symbol)
    if d == :R
        return getfield(QR,:R)
    elseif d == :Q
        return BandedQ(QR.H,size(QR,1))
    else
        getfield(QR, d)
    end
end
# iteration for destructuring into components
iterate(S::BandedQR) = (S.Q, Val(:R))
iterate(S::BandedQR, ::Val{:R}) = (S.R, Val(:done))
iterate(S::BandedQR, ::Val{:done}) = nothing

function qr(A::BandedMatrix)
    R=BandedMatrix(Zeros{eltype(A)}(size(A)), (A.l,A.l+A.u))
    R.data[A.l+1:end,:]=A.data
    banded_qr!(R)
end

flipsign(x,y) = Base.flipsign(x,y)
flipsign(x::BigFloat,y::BigFloat) = sign(y)==1 ? x : (-x)
flipsign(x,y::Complex) = y==0 ? x : x*sign(y)

function banded_qr!(R::BandedMatrix{T}) where T
    M=R.l+1   # number of diag+subdiagonal bands
    m,n=size(R)
    W=Matrix{T}(undef, M, (n<m ? n : m-1))
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
            axpy!(M,-2*dt,wp,1,v,1)
        end
    end

    for k=m-R.l+1:(n<m ? n : m-1)
        p=k-m+R.l
        v=r+sz*(R.u + (k-1)*st)    # diagonal entry
        wp=w+stw*sz*(k-1)          # k-th column of W
        BLAS.blascopy!(M-p,v,1,wp,1)
        W[1,k]+= flipsign(BLAS.nrm2(M-p,wp,1),W[1,k])
        normalize!(M-p,wp)

        for j=k:min(k+R.u,n)
            v=r+sz*(R.u + (k-1)*st + (j-k)*(st-1))
            dt=dot(M-p,wp,1,v,1)
            axpy!(M-p,-2*dt,wp,1,v,1)
        end
    end

    BandedQR(W, _BandedMatrix(R.data[1:R.u+1,:],m,0,R.u))
end
