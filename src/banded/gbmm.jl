gbmv!{T<:BlasFloat}(trans::Char, α::T, A::AbstractMatrix{T}, x::StridedVector{T}, β::T, y::StridedVector{T}) =
    gbmv!(trans,size(A,1), size(A,2), bandwidth(A,1), bandwidth(A,2), α,
          pointer(A), leadingdimension(A), pointer(x), stride(x,1), β, pointer(y), stride(y,1))


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

   gbmv!('N',min(Cl+j,n), min(Bl+j,ν),
          Al, Au,
          α,
          a, sta,
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

   gbmv!('N', min(Cl+j,n), min(Bl+Bu+1,ν-j+Bu+1),
           Al+j-Bu-1, Au-j+Bu+1,
           α,
           a+sz*(j-Bu-1)*sta, sta,
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
   gbmv!('N', min(Cl+Cu+1,n-j+Cu+1), min(Bl+Bu+1,ν-p+1),
           Al+Au, 0,
           α,
           a+sz*(j-Bu-1)*sta, sta,
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

function gbmm!{T<:BlasFloat}(tA::Char, tB::Char, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T})
    if tA ≠ 'N' || tB ≠ 'N'
        error("Only 'N' flag is supported.")
    end
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
    # j ≤ ν + B.u since then 1+p ≤ ν, so inside the columns of A

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

function gbmm!{T<:BlasFloat}(tA::Char, tB::Char, α::T, A::AbstractMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T})
    if tA ≠ 'N' || tB ≠ 'N'
        error("Only 'N' flag is supported.")
    end
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
        gbmv!('N',n,ν,Al,Au,α,a,st,b+(j-1)*sz*stb,stride(B,1),β,c+(j-1)*sz*stc,stride(C,1))
    end
    C
end

function gbmm!{T<:BlasFloat}(tA::Char, tB::Char, α::T, A::StridedMatrix{T}, B::AbstractMatrix{T}, β::T, C::StridedMatrix{T})
    if tA ≠ 'N' || tB ≠ 'N'
        error("Only 'N' flag is supported.")
    end
    st = leadingdimension(B)
    n,ν = size(B)
    a = pointer(A)
    b = pointer(B)
    m = size(A, 1)

    @assert size(C,1) == m
    @assert size(C,2) == ν

    c=pointer(C)
    sz=sizeof(T)

    Bl = bandwidth(B,1); Bu = bandwidth(B,2)

    for j=1:m
        gbmv!('T',n,ν,Bl,Bu,α,b,st,a+(j-1)*sz,stride(A,2),β,c+(j-1)*sz,stride(C,2))
    end
    C
end

αA_mul_B_plus_βC!{T}(α,A::BLASBandedMatrix{T},x,β,y) = gbmv!('N',α,A,x,β,y)
αA_mul_B_plus_βC!(α,A::StridedMatrix,x,β,y) = BLAS.gemv!('N',α,A,x,β,y)
αA_mul_B_plus_βC!(α,A,x,β,y) = (y .= α*A*x + β*y)
αA_mul_B_plus_βC!{T,U,V}(α,A::BLASBandedMatrix{T},B::BLASBandedMatrix{U},β,C::BLASBandedMatrix{V}) = gbmm!(α,A,B,β,C)
αA_mul_B_plus_βC!(α,A::StridedMatrix,B::StridedMatrix,β,C::StridedMatrix) = BLAS.gemm!('N','N',α,A,B,β,C)
