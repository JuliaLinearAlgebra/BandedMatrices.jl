## These routines give access to the necessary information to call BLAS


@inline leadingdimension(B::BandedMatrix) = stride(B.data,2)
@inline leadingdimension{T}(B::BandedSubMatrix{T}) = leadingdimension(parent(B))


@inline Base.pointer(B::BandedMatrix) = pointer(B.data)
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{Colon,Colon}}) =
    pointer(parent(B))
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},Colon}}) =
    pointer(parent(B))
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{Colon,UnitRange{Int}}}) =
    pointer(parent(B))+leadingdimension(parent(B))*(first(parentindexes(B)[2])-1)*sizeof(T)
@inline Base.pointer{T}(B::SubArray{T,2,BandedMatrix{T},Tuple{UnitRange{Int},UnitRange{Int}}}) =
    pointer(parent(B))+leadingdimension(parent(B))*(first(parentindexes(B)[2])-1)*sizeof(T)





# We implement for pointers



if VERSION < v"0.5.0-dev"
    macro blasfunc(x)
       return :( $(BLAS.blasfunc(x) ))
    end
else
    import Base.BLAS.@blasfunc
end



function Base.scale!(α::Number, A::BandedMatrix)
    Base.scale!(α, A.data)
    A
end

function Base.scale!(A::BandedMatrix, α::Number)
    Base.scale!(A.data, α)
    A
end




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


gbmv!{T<:BlasFloat}(trans::Char,α::T,A::BandedMatrix{T},x::StridedVector{T},β::T,y::StridedVector{T}) =
    gbmv!(trans,A.m,A.l,A.u,α,A.data,x,β,y)

gbmv!{T<:BlasFloat}(trans::Char,α::T,A::AbstractMatrix{T},x::StridedVector{T},β::T,y::StridedVector{T}) =
    gbmv!(trans,size(A,1),bandwidth(A,1),bandwidth(A,2),α,
          pointer(A),size(A,2),leadingdimension(A),
          pointer(x),stride(x,1),β,pointer(y),stride(y,1))





# this is matrix*matrix
gbmm!{U,V,T}(α,A::AbstractMatrix{U},B::AbstractMatrix{V},β,C::AbstractMatrix{T}) =
    gbmm!(convert(T,α),convert(AbstractMatrix{T},A),convert(AbstractMatrix{T},B),
          convert(T,β),C)


αA_mul_B_plus_βC!{T}(α,A::BLASBandedMatrix{T},x,β,y) = gbmv!('N',α,A,x,β,y)
αA_mul_B_plus_βC!(α,A::StridedMatrix,x,β,y) = BLAS.gemv!('N',α,A,x,β,y)

αA_mul_B_plus_βC!(α,A,x,β,y) = (y .= α*A*x + β*y)


αA_mul_B_plus_βC!{T,U,V}(α,A::BLASBandedMatrix{T},B::BLASBandedMatrix{U},β,C::BLASBandedMatrix{V}) =
    gbmm!(α,A,B,β,C)
αA_mul_B_plus_βC!(α,A::StridedMatrix,B::StridedMatrix,β,C::StridedMatrix) = BLAS.gemm!('N','N',α,A,B,β,C)


# The following routines multiply
#
#  C[:,j] = α*A*B[:,j] + β* C[:,j]
#
#  But are divided into cases to minimize the calculated
#  rows.  We use A is n x ν, B is ν x m



# Equivalent to
#
#  C[1:min(Cl+j,n),j] = α*A[1:min(Cl+j,n),1:min(Bl+j,ν)]*B[1:min(Bl+j,ν),j]
#                       + β*C[1:min(Cl+j,n),j]
#
@inline function A11_Btop_Ctop_gbmv!(α,β,
                               n,ν,m,j,
                               sz,
                               a,Al,Au,sta,
                               b,Bl,Bu,stb,
                               c,Cl,Cu,stc)
   # A=BandedMatrix(pointer_to_array(a,(Al+Au+1,ν)),n,Al,Au)
   # B=BandedMatrix(pointer_to_array(b,(Bl+Bu+1,m)),ν,Bl,Bu)
   # C=BandedMatrix(pointer_to_array(c,(Cl+Cu+1,m)),n,Cl,Cu)
   #
   # nr=1:min(Cl+j,n)
   # νr=1:min(Bl+j,ν)
   #
   # cj = α*A[nr,νr]*B[νr,j] + β*C[nr,j]
   #
   # for k in nr
   #     C[k,j]=cj[k-first(nr)+1]
   # end
   #
   # c

   gbmv!('N',min(Cl+j,n),
          Al, Au,
          α,
          a, min(Bl+j,ν), sta,
          b+sz*((j-1)*stb+Bu-j+1), 1, β,
          c+sz*((j-1)*stc+Cu-j+1), 1)
end


# Equivalent to
#
#  C[1:min(Cl+j,n),j] = α*A[1:min(Cl+j,n),p:min(p+Bl+Bu+1,ν)]*
#                               B[p:min(p+Bl+Bu+1,ν),j]
#                       + β*C[1:min(Cl+j,n),j]
# for p = j-B.u
#


@inline function Atop_Bmid_Ctop_gbmv!(α,β,
                               n,ν,m,j,
                               sz,
                               a,Al,Au,sta,
                               b,Bl,Bu,stb,
                               c,Cl,Cu,stc)
   # p=j-Bu

   # A=BandedMatrix(pointer_to_array(a,(Al+Au+1,ν)),n,Al,Au)
   # B=BandedMatrix(pointer_to_array(b,(Bl+Bu+1,m)),ν,Bl,Bu)
   # C=BandedMatrix(pointer_to_array(c,(Cl+Cu+1,m)),n,Cl,Cu)
   #
   # nr=1:min(Cl+j,n)
   # νr=p:min(p+Bl+Bu+1,ν)
   #
   # cj = α*A[nr,νr]*B[νr,j] + β*C[nr,j]
   #
   # for k in nr
   #     C[k,j]=cj[k-first(nr)+1]
   # end
   #
   # c

   gbmv!('N', min(Cl+j,n),
           Al+j-Bu-1, Au-j+Bu+1,
           α,
           a+sz*(j-Bu-1)*sta, min(Bl+Bu+1,ν-j+Bu+1), sta,
           b+sz*(j-1)*stb, 1, β,
           c+sz*((j-1)*stc+Cu-j+1), 1)
end


# Equivalent to
#
#  C[nr,j] = α*A[nr,νr]*B[νr,j] + β*C[nr,j]
#
# for p  = j-B.u
#     nr = p-Au : min(p+Al,n)
#     νr = p    : min(p+Bl+Bu+1,ν)
#
#

@inline function Amid_Bmid_Cmid_gbmv!(α,β,
                               n,ν,m,j,
                               sz,
                               a,Al,Au,sta,
                               b,Bl,Bu,stb,
                               c,Cl,Cu,stc)
   p = j-Bu
   # A = BandedMatrix(pointer_to_array(a,(Al+Au+1,ν)),n,Al,Au)
   # B = BandedMatrix(pointer_to_array(b,(Bl+Bu+1,m)),ν,Bl,Bu)
   # C = BandedMatrix(pointer_to_array(c,(Cl+Cu+1,m)),n,Cl,Cu)
   #
   # nr= j-Cu : min(j+Cl,n)
   # νr= p    : min(p+Bl+Bu+1,ν)
   # if !isempty(nr) && !isempty(νr)
   #     cj = α*A[nr,νr]*B[νr,j] + β*C[nr,j]
   #
   #     for k in nr
   #         C[k,j]=cj[k-first(nr)+1]
   #     end
   # end
   #
   # c
   gbmv!('N', min(Cl+Cu+1,n-j+Cu+1),
           Al+Au, 0,
           α,
           a+sz*(j-Bu-1)*sta, min(Bl+Bu+1,ν-p+1), sta,
           b+sz*(j-1)*stb, 1, β,
           c+sz*(j-1)*stc, 1)
end

# Equivalent to
#
#  C[nr,j] =  β*C[nr,j]
#
# for nr= max(1,j-Cu) : min(j+Cl,n)
#


@inline function Anon_Bnon_C_gbmv!(α,β,
                               n,ν,m,j,
                               sz,
                               a,Al,Au,sta,
                               b,Bl,Bu,stb,
                               c,Cl,Cu,stc)
   # C = BandedMatrix(pointer_to_array(c,(Cl+Cu+1,m)),n,Cl,Cu)
   #
   # nr= max(1,j-Cu) : min(j+Cl,n)
   #
   # if !isempty(nr)
   #     cj =  β*C[nr,j]
   #
   #     for k in nr
   #         C[k,j]=cj[k-first(nr)+1]
   #     end
   # end
   #
   # c

   BLAS.scal!(Cu+Cl+1,β,c+sz*(j-1)*stc,1)
end



# function Amid_Bbot_Cmid_gbmv!(α,β,
#                                n,ν,m,j,
#                                sz,
#                                a,Al,Au,sta,
#                                b,Bl,Bu,stb,
#                                c,Cl,Cu,stc)
#    gbmv!('N', Cl+Cu+1,
#            Al+Au, 0,
#            α,
#            a+sz*(j-Bu-1)*sta, ν-j+1, sta,
#            b+sz*(j-1)*stb, β,
#            c+sz*(j-1)*stc)
# end
#
#
# function Abot_Bbot_Cbot_gbmv!(α,β,
#                                n,ν,m,j,
#                                sz,
#                                a,Al,Au,sta,
#                                b,Bl,Bu,stb,
#                                c,Cl,Cu,stc)
#    gbmv!('N', n-j+C.u+1,
#            A.l+A.u, 0,
#            α,
#            a+sz*(j-B.u-1)*sta, B.l+B.u+1-(j-ν+B.l), sta,
#            b+sz*(j-1)*stb, β,
#            c+sz*(j-1)*stc)
# end



function gbmm!{T<:BlasFloat}(α::T,A::AbstractMatrix{T},B::AbstractMatrix{T},β::T,C::AbstractMatrix{T})
    n,ν = size(A)
    m = size(B,2)

    @assert n == size(C,1)
    @assert ν == size(B,1)
    @assert m == size(C,2)

    Al = bandwidth(A,1); Au = bandwidth(A,2)
    Bl = bandwidth(B,1); Bu = bandwidth(B,2)
    Cl = bandwidth(C,1); Cu = bandwidth(C,2)

    # only tested at the moment for this case
    # TODO: implement when C.u,C.l ≥
    @assert Cu == Au+Bu
    @assert Cl == Al+Bl


    a = pointer(A)
    b = pointer(B)
    c = pointer(C)
    sta = leadingdimension(A)
    stb = leadingdimension(B)
    stc = leadingdimension(C)
    sz = sizeof(T)



    # Multiply columns j where B[1,j]≠0: A is at 1,1 and C[1,j]≠0
    for j = 1:min(m,1+Bu)
        A11_Btop_Ctop_gbmv!(α,β,
                            n,ν,m,j,
                            sz,
                            a,Al,Au,sta,
                            b,Bl,Bu,stb,
                            c,Cl,Cu,stc)
    end

    # Multiply columns j where B[k,j]=0 for k<p=(j-B.u-1), A is at 1,1+p and C[1,j]≠0
    # j ≤ ν + B.u since then 1+p ≤ ν, so inside the columns of A

    for j = 2+Bu:min(1+Cu,ν+Bu,m)
        Atop_Bmid_Ctop_gbmv!(α,β,
                            n,ν,m,j,
                            sz,
                            a,Al,Au,sta,
                            b,Bl,Bu,stb,
                            c,Cl,Cu,stc)
    end


    # multiply columns where A and B are mid and C is bottom
    for j = 2+Cu:min(m,ν+Bu,n+Cu)
        Amid_Bmid_Cmid_gbmv!(α,β,
                            n,ν,m,j,
                            sz,
                            a,Al,Au,sta,
                            b,Bl,Bu,stb,
                            c,Cl,Cu,stc)
    end

    # scale columns of C by β that aren't impacted by α*A*B
    for j = ν+Bu+1:min(m,n+Cu)
        Anon_Bnon_C_gbmv!(α,β,
                            n,ν,m,j,
                            sz,
                            a,Al,Au,sta,
                            b,Bl,Bu,stb,
                            c,Cl,Cu,stc)
    end

    C
end

#TODO: Speedup
function gbmm!{T}(α::T,A::AbstractMatrix{T},B::AbstractMatrix{T},β::T,C::AbstractMatrix{T})
    for j=1:size(C,2),k=colrange(C,j)
        C[k,j]*=β
    end

    m=size(C,2)
    Bl = bandwidth(B,1); Bu = bandwidth(B,2)
    for ν=1:size(A,2),k=colrange(A,ν)
        for j=max(ν-Bl,1):min(ν+Bu,m)
            C[k,j]+=α*A[k,ν]*B[ν,j]
        end
    end

    C
end


function gbmm!{T<:BlasFloat}(α::T,A::AbstractMatrix{T},B::StridedMatrix{T},β::T,C::StridedMatrix{T})
    st = leadingdimension(A)
    n,ν = size(A)
    a = pointer(A)


    b = pointer(B)
    stb = stride(B,2)


    m = size(B,2)

    @assert size(C,1) == n
    @assert size(C,2) == m

    c=pointer(C)
    stc=stride(C,2)
    sz=sizeof(T)

    Al = bandwidth(A,1); Au = bandwidth(A,2)

    for j=1:m
        gbmv!('N',n,Al,Au,α,a,ν,st,b+(j-1)*sz*stb,stride(B,1),β,c+(j-1)*sz*stc,stride(C,1))
    end
    C
end

function banded_axpy!(a::Number,X,Y)
    n,m = size(X)
    if (n,m) ≠ size(Y)
        throw(BoundsError())
    end
    Xl,Xu = bandwidths(X)
    Yl,Yu = bandwidths(Y)

    if Xl > Yl
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j+Yl+1):min(j+Xl,n)
            if inbands_getindex(X,k,j) ≠ 0
                error("X has nonzero entries in bands outside bandrange of Y.")
            end
        end
    end
    if Xu > Yu
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j-Xu):min(j-Yu-1,n)
            if inbands_getindex(X,k,j) ≠ 0
                error("X has nonzero entries in bands outside bandrange of Y.")
            end
        end
    end

    l = min(Xl,Yl)
    u = min(Xu,Yu)

    @inbounds for j=1:m,k=max(1,j-u):min(n,j+l)
        inbands_setindex!(Y,a*inbands_getindex(X,k,j)+inbands_getindex(Y,k,j),k,j)
    end
    Y
end


function banded_axpy!{T}(a::Number,X,S::BandedSubMatrix{T})
    @assert size(X)==size(S)

    Y=parent(S)
    kr,jr=parentindexes(S)

    if isempty(kr) || isempty(jr)
        return S
    end

    shft=bandshift(S)

    @assert bandwidth(X,2) ≥ -bandwidth(X,1)

    if bandwidth(X,1) > bandwidth(Y,1)-shft
        bS = bandwidth(Y,1)-shft
        bX = bandwidth(X,1)
        for j=1:size(X,2),k=max(1,j+bS+1):min(j+bX,size(X,1))
            if X[k,j] ≠ 0
                error("Cannot add banded matrix to matrix with smaller bandwidth: entry $k,$j is $(X[k,j]).")
            end
        end
    end

    if bandwidth(X,2) > bandwidth(Y,2)+shft
        bS = bandwidth(Y,2)+shft
        bX = bandwidth(X,2)
        for j=1:size(X,2),k=max(1,j-bX):min(j-bS-1,size(X,1))
            if X[k,j] ≠ 0
                error("Cannot add banded matrix to matrix with smaller bandwidth: entry $k,$j is $(X[k,j]).")
            end
        end
    end


    for j=1:size(X,2),k=colrange(X,j)
        @inbounds Y.data[kr[k]-jr[j]+Y.u+1,jr[j]]+=a*inbands_getindex(X,k,j)
    end

    S
end


if VERSION < v"0.5"
    # taken from 0.5
    function checksquare(A)
        m,n = size(A)
        m == n || throw(DimensionMismatch("matrix is not square"))
        m
    end
else
    checksquare(A) = LinAlg.checksquare(A)
end


function Base.BLAS.axpy!{T}(a::Number,X::UniformScaling,Y::BLASBandedMatrix{T})
    checksquare(Y)

    α = a*X.λ
    for k=1:size(Y,1)
        @inbounds Y[k,k] += α
    end
    Y
end

Base.BLAS.axpy!(a::Number,X::BandedMatrix,Y::BandedMatrix) =
    banded_axpy!(a,X,Y)

Base.BLAS.axpy!{T}(a::Number,X::BandedMatrix,S::BandedSubMatrix{T}) =
    banded_axpy!(a,X,S)

function Base.BLAS.axpy!{T1,T2}(a::Number,X::BandedSubMatrix{T1},Y::BandedSubMatrix{T2})
    if bandwidth(X,1) < 0
        jr=1-bandwidth(X,1):size(X,2)
        banded_axpy!(a,view(X,:,jr),view(Y,:,jr))
    elseif bandwidth(X,2) < 0
        kr=1-bandwidth(X,2):size(X,1)
        banded_axpy!(a,view(X,kr,:),view(Y,kr,:))
    else
        banded_axpy!(a,X,Y)
    end
end

function Base.BLAS.axpy!{T}(a::Number,X::BandedSubMatrix{T},Y::BandedMatrix)
    if bandwidth(X,1) < 0
        jr=1-bandwidth(X,1):size(X,2)
        banded_axpy!(a,view(X,:,jr),view(Y,:,jr))
    elseif bandwidth(X,2) < 0
        kr=1-bandwidth(X,2):size(X,1)
        banded_axpy!(a,view(X,kr,:),view(Y,kr,:))
    else
        banded_axpy!(a,X,Y)
    end
end


# used to add a banded matrix to a dense matrix
function banded_dense_axpy!(a,X,Y)
    @assert size(X)==size(Y)
    @inbounds for j=1:size(X,2),k=colrange(X,j)
        Y[k,j]+=a*inbands_getindex(X,k,j)
    end
    Y
end

Base.BLAS.axpy!(a::Number,X::BandedMatrix,Y::AbstractMatrix) =
    banded_dense_axpy!(a,X,Y)






## A_mul_B! overrides

Base.A_mul_B!{T}(C::Matrix,A::BLASBandedMatrix{T},B::StridedMatrix) =
    gbmm!(one(eltype(C)),A,B,zero(eltype(C)),C)

## Matrix*Vector Multiplicaiton



function banded_A_mul_B!{T<:BlasFloat}(c::AbstractVector{T},A::AbstractMatrix{T},b::StridedVector{T})
    m,n = size(A)

    @boundscheck if length(c) ≠ m || length(b) ≠ n
        throw(DimensionMismatch())
    end

    l,u = bandwidths(A)
    if l < 0 && u < 0
        # no bands
        c[:] = zero(eltype(c))
    elseif l < 0
        A_mul_B!(c,view(A,:,1-l:n),view(b,1-l:n))
    elseif u < 0
        c[1:-u] = zero(eltype(c))
        A_mul_B!(view(c,1-u:m),view(A,1-u:m,:),b)
    else
        gbmv!('N',m,l,u,one(T),
                pointer(A),n,leadingdimension(A),pointer(b),stride(b,1),zero(T),pointer(c),stride(c,1))
    end
    c
end


Base.A_mul_B!{T}(c::AbstractVector,A::BLASBandedMatrix{T},b::AbstractVector) =
    banded_A_mul_B!(c,A,b)






## Matrix*Matrix Multiplication


function Base.A_mul_B!{T,U,V}(C::BLASBandedMatrix{T},A::BLASBandedMatrix{U},B::BLASBandedMatrix{V})
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    Am, An = size(A)
    Bm, Bn = size(B)

    if (Al < 0 && Au < 0) || (Bl < 0 && Bu < 0)
        C.data[:,:] = zero(T)
    elseif Al < 0
        A_mul_B!(C,view(A,:,1-Al:An),view(B,1-Al:An,:))
    elseif Au < 0
        C[1:-Au,:] = zero(T)
        A_mul_B!(view(C,1-Au:Am,:),view(A,1-Au:Am,:),B)
    elseif Bl < 0
        C[:,1:-Bl] = zero(T)
        A_mul_B!(view(C,:,1-Bl:Bn),A,view(B,:,1-Bl:Bn))
    elseif Bu < 0
        A_mul_B!(C,view(A,:,1-Bu:Bm),view(B,1-Bu:Bm,:))
    else
        gbmm!(one(eltype(A)),A,B,zero(eltype(C)),C)
    end
    C
end



## Method definitions for generic eltypes - will make copies

# Direct and transposed algorithms
for typ in [BandedMatrix, BandedLU]
    for fun in [:A_ldiv_B!, :At_ldiv_B!]
        @eval function $fun{T<:Number, S<:Number}(A::$typ{T}, B::StridedVecOrMat{S})
            checksquare(A)
            AA, BB = _convert_to_blas_type(A, B)
            $fun(lufact(AA), BB) # call BlasFloat versions
        end
    end
    # \ is different because it needs a copy, but we have to avoid ambiguity
    @eval function (\){T<:BlasReal}(A::$typ{T}, B::VecOrMat{Complex{T}})
        checksquare(A)
        A_ldiv_B!(convert($typ{Complex{T}}, A), copy(B)) # goes to BlasFloat call
    end
    @eval function (\){T<:Number, S<:Number}(A::$typ{T}, B::StridedVecOrMat{S})
        checksquare(A)
        TS = _promote_to_blas_type(T, S)
        A_ldiv_B!(convert($typ{TS}, A), copy_oftype(B, TS)) # goes to BlasFloat call
    end
end

# Hermitian conjugate
for typ in [BandedMatrix, BandedLU]
    @eval function Ac_ldiv_B!{T<:Complex, S<:Number}(A::$typ{T}, B::StridedVecOrMat{S})
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        Ac_ldiv_B!(lufact(AA), BB) # call BlasFloat versions
    end
    @eval Ac_ldiv_B!{T<:Real, S<:Real}(A::$typ{T}, B::StridedVecOrMat{S}) =
        At_ldiv_B!(A, B)
    @eval Ac_ldiv_B!{T<:Real, S<:Complex}(A::$typ{T}, B::StridedVecOrMat{S}) =
        At_ldiv_B!(A, B)
end


# Method definitions for BlasFloat types - no copies

# Direct and transposed algorithms
for (ch, fname) in zip(('N', 'T'), (:A_ldiv_B!, :At_ldiv_B!))
    # provide A*_ldiv_B!(::BandedLU, ::StridedVecOrMat) for performance
    @eval function $fname{T<:BlasFloat}(A::BandedLU{T}, B::StridedVecOrMat{T})
        checksquare(A)
        gbtrs!($ch, A.l, A.u, A.m, A.data, A.ipiv, B)
    end
    # provide A*_ldiv_B!(::BandedMatrix, ::StridedVecOrMat) for generality
    @eval function $fname{T<:BlasFloat}(A::BandedMatrix{T}, B::StridedVecOrMat{T})
        checksquare(A)
        $fname(lufact(A), B)
    end
end

# Hermitian conjugate algorithms - same two routines as above
function Ac_ldiv_B!{T<:BlasComplex}(A::BandedLU{T}, B::StridedVecOrMat{T})
    checksquare(A)
    gbtrs!('C', A.l, A.u, A.m, A.data, A.ipiv, B)
end

function Ac_ldiv_B!{T<:BlasComplex}(A::BandedMatrix{T}, B::StridedVecOrMat{T})
    checksquare(A)
    Ac_ldiv_B!(lufact(A), B)
end

# fall back for real inputs
Ac_ldiv_B!{T<:BlasReal}(A::BandedLU{T}, B::StridedVecOrMat{T}) =
    At_ldiv_B!(A, B)
Ac_ldiv_B!{T<:BlasReal}(A::BandedMatrix{T}, B::StridedVecOrMat{T}) =
    At_ldiv_B!(A, B)
