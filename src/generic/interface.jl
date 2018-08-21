



# some default functions




# matrix * matrix

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'C'}, tB::Val{'N'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,ν,k)' * inbands_getindex(B,ν,j)
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'N'}, tB::Val{'C'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)


    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,j,ν)'
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'C'}, tB::Val{'C'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)


    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,ν,k)' * inbands_getindex(B,j,ν)'
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'T'}, tB::Val{'N'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + transpose(inbands_getindex(A,ν,k)) * inbands_getindex(B,ν,j)
        end
        inbands_setindex!(C,tmp,k,j)
    end

    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'N'}, tB::Val{'T'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,k,ν) * transpose(inbands_getindex(B,j,ν))
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'T'}, tB::Val{'T'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + transpose(inbands_getindex(A,ν,k)) * transpose(inbands_getindex(B,j,ν))
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'C'}, tB::Val{'T'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,ν,k)' * transpose(inbands_getindex(B,j,ν))
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'T'}, tB::Val{'C'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(An,k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + transpose(inbands_getindex(A,ν,k)) * inbands_getindex(B,j,ν)'
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end

# use BLAS routine for positively banded BlasBandedMatrices
function _positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Val, tB::Val, A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                       ::BandedColumnMajor, ::BandedColumnMajor, ::BandedColumnMajor) where {T}
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    _banded_generic_matmatmul!(C, tA, tB, A, B)
end

function _positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Val{'N'}, tB::Val{'N'}, A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                       ::BandedColumnMajor, ::BandedColumnMajor, ::BandedColumnMajor) where {T}
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    # _banded_generic_matmatmul! is faster for sparse matrix
    if (Al + Au < 100 && Bl + Bu < 100)
        _banded_generic_matmatmul!(C, tA, tB, A, B)
    else
        # TODO: implement gbmm! routines for other flags
        gbmm!('N', 'N', one(T), A, B, zero(T), C)
    end
end



function generally_banded_matmatmul!(C::AbstractMatrix{T}, tA::Val, tB::Val, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    if An != Bm || size(C, 1) != Am || size(C, 2) != Bn
        throw(DimensionMismatch("*"))
    end
    # TODO: checkbandmatch

    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)

    if (-Al > Au) || (-Bl > Bu)   # A or B has empty bands
        fill!(C, zero(T))
    elseif Al < 0
        C[max(1,Bn+Al-1):Am, :] .= zero(T)
        banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Al:An), _view(tB, B, 1-Al:An, :))
    elseif Au < 0
        C[1:-Au,:] .= zero(T)
        banded_matmatmul!(view(C, 1-Au:Am,:), tA, tB, _view(tA, A, 1-Au:Am,:), B)
    elseif Bl < 0
        C[:, 1:-Bl] .= zero(T)
        banded_matmatmul!(view(C, :, 1-Bl:Bn), tA, tB, A, _view(tB, B, :, 1-Bl:Bn))
    elseif Bu < 0
        C[:, max(1,Am+Bu-1):Bn] .= zero(T)
        banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Bu:Bm), _view(tB, B, 1-Bu:Bm, :))
    else
        positively_banded_matmatmul!(C, tA, tB, A, B)
    end
    C
end


banded_generic_matmatmul!(C::AbstractMatrix, tA::Val, tB::Val, A::AbstractMatrix, B::AbstractMatrix) =
    generally_banded_matmatmul!(C, tA, tB, A, B)


macro banded(Typ)
    ret = quote
        BandedMatrices.MemoryLayout(A::$Typ) = BandedMatrices.BandedColumnMajor()
        BandedMatrices.isbanded(::$Typ) = true
    end
    esc(ret)
end
