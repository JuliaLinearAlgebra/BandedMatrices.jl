


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

function _num_zeroband_u(A)
    Al, Au = bandwidths(A)
    for b = 0:Al+Au
        any(!iszero, view(A,band(Au-b))) && return b
    end
    return Al+Au+1
end

function _num_zeroband_l(A)
    Al, Au = bandwidths(A)
    for b = 0:Al+Au
        any(!iszero, view(A,band(-Al+b))) && return b
    end
    return Al+Au+1
end

function gbmm!(tA::Char, tB::Char, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T<:BlasFloat}
    if tA ≠ 'N' || tB ≠ 'N'
        error("Only 'N' flag is supported.")
    end
    n,ν = size(A)
    m = size(B,2)

    @assert n == size(C,1)
    @assert ν == size(B,1)
    @assert m == size(C,2)

    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    C̃l, C̃u = bandwidths(C)
    Cl,Cu = Al+Bl,Au+Bu

    # prune zero bands
    if C̃u < Cu
        Au_r, Bu_r = _num_zeroband_u(A), _num_zeroband_u(B)
        C̃u ≥  Cu - Au_r - Bu_r || throw(BandError(C, Cu - Au_r - Bu_r))
        A_data = bandeddata(A)
        B_data = bandeddata(B)

        if Au-Au_r < -Al || Bu - Bu_r < -Bl
            _fill_lmul!(β, C)
            return C
        end

        Ã = _BandedMatrix(@views(A_data[Au_r+1:end,:]), n, Al, Au-Au_r)
        B̃ = _BandedMatrix(@views(B_data[Bu_r+1:end,:]), ν, Bl, Bu-Bu_r)
        return materialize!(MulAdd(α, Ã, B̃, β, C))
    end

    if C̃l < Cl
        Al_r, Bl_r = _num_zeroband_l(A), _num_zeroband_l(B)
        C̃l ≥  Cl - Al_r - Bl_r || throw(BandError(C, Cl - Al_r - Bl_r))
        A_data = bandeddata(A)
        B_data = bandeddata(B)

        if Al-Al_r < -Au || Bl - Bl_r < -Bu
            _fill_lmul!(β, C)
            return C
        end

        Ã = _BandedMatrix(@views(A_data[1:end-Al_r,:]), n, Al-Al_r, Au)
        B̃ = _BandedMatrix(@views(B_data[1:end-Bl_r,:]), ν, Bl-Bl_r, Bu)
        return materialize!(MulAdd(α, Ã, B̃, β, C))
    end

    A_data = bandeddata(A)
    B_data = bandeddata(B)
    C̃_data = bandeddata(C)

    # scale extra bands
    _fill_lmul!(β, view(C̃_data, 1:C̃u-Cu,:))
    _fill_lmul!(β, view(C̃_data, (C̃u+Cl+1)+1:size(C̃_data,1),:))
    C_data = view(C̃_data, (C̃u-Cu+1):(C̃u+Cl+1), :) # shift to bands we will write to
    _gbmm!(α, A_data, B_data, β, C_data, (n,ν,m), (Al, Au), (Bl, Bu), (Cl, Cu))

    C
end


function _gbmm!(α::T, A_data, B_data, β, C_data, (n,ν,m), (Al, Au), (Bl, Bu), (Cl, Cu)) where T
    a = pointer(A_data)
    b = pointer(B_data)
    c = pointer(C_data)
    sta = stride(A_data,2)
    stb = stride(B_data,2)
    stc = stride(C_data,2)
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
    _fill_lmul!(β, view(C_data, :, ν+Bu+1:min(m,n+Cu)))
end
