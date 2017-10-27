# some functions

function banded_axpy! end
function banded_matmatmul! end
function banded_matvecmul! end


# additions and subtractions

@propagate_inbounds function banded_generic_axpy!(a::Number, X::AbstractMatrix{U}, Y::AbstractMatrix{V}) where {U, V}
    n,m = size(X)
    if (n,m) ≠ size(Y)
        throw(BoundsError())
    end
    Xl, Xu = bandwidths(X)
    Yl, Yu = bandwidths(Y)

    @boundscheck if Xl > Yl
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j+Yl+1):min(j+Xl,n)
            if inbands_getindex(X,k,j) ≠ 0
                throw(BandError(X, (k,j)))
            end
        end
    end
    @boundscheck if Xu > Yu
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j-Xu):min(j-Yu-1,n)
            if inbands_getindex(X,k,j) ≠ 0
                throw(BandError(X, (k,j)))
            end
        end
    end

    l = min(Xl,Yl)
    u = min(Xu,Yu)

    @inbounds for j=1:m,k=max(1,j-u):min(n,j+l)
        inbands_setindex!(Y, a*inbands_getindex(X,k,j) + inbands_getindex(Y,k,j) ,k, j)
    end
    Y
end

function banded_dense_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    if size(X) != size(Y)
        throw(DimensionMismatch("+"))
    end
    @inbounds for j=1:size(X,2),k=colrange(X,j)
        Y[k,j] += a*inbands_getindex(X,k,j)
    end
    Y
end


# matrix * vector

function _banded_generic_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V}
    @inbounds c[:] = zero(T)
    if tA == 'N'
        @inbounds for j = 1:size(A,2), k = colrange(A,j)
            c[k] += inbands_getindex(A,k,j)*b[j]
        end
    elseif tA == 'C'
        @inbounds for j = 1:size(A,2), k = colrange(A,j)
            c[j] += inbands_getindex(A,k,j)'*b[k]
        end
    elseif tA == 'T'
        @inbounds for j = 1:size(A,2), k = colrange(A,j)
            c[j] += inbands_getindex(A,k,j)*b[k]
        end
    end
    c
end


positively_banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V} =
    _banded_generic_matvecmul!(c, tA, A, b)
positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V} =
    _banded_generic_matmatmul!(C, tA, tB, A, B)

function generally_banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V}
    m, n = _size(tA, A)
    if length(c) ≠ m || length(b) ≠ n
        throw(DimensionMismatch("*"))
    end

    l, u = _bandwidths(tA, A)
    if -l > u
        # no bands
        c[:] = zero(T)
    elseif l < 0
        banded_matvecmul!(c, tA, _view(tA, A, :, 1-l:n), view(b, 1-l:n))
    elseif u < 0
        c[1:-u] = zero(T)
        banded_matvecmul!(view(c, 1-u:m), tA, _view(tA, A, 1-u:m, :), b)
    else
        positively_banded_matvecmul!(c, tA, A, b)
    end
    c
end

banded_generic_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V} = generally_banded_matvecmul!(c, tA, A, b)


# matrix * matrix
function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    Cl,Cu = prodbandwidths(tA, tB, A, B)

    if tA == 'N' && tB == 'N'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,ν,j)
            end
            inbands_setindex!(C,tmp,k,j)
        end
    elseif tA == 'C' && tB == 'N'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,ν,k)' * inbands_getindex(B,ν,j)
            end
            inbands_setindex!(C,tmp,k,j)
        end
    elseif tA == 'N' && tB == 'C'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,j,ν)'
            end
            inbands_setindex!(C,tmp,k,j)
        end
    elseif tA == 'C' && tB == 'C'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,ν,k)' * inbands_getindex(B,j,ν)'
            end
            inbands_setindex!(C,tmp,k,j)
        end
    elseif tA == 'T' && tB == 'N'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,ν,k) * inbands_getindex(B,ν,j)
            end
            inbands_setindex!(C,tmp,k,j)
        end

    elseif tA == 'N' && tB == 'T'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,j,ν)
            end
            inbands_setindex!(C,tmp,k,j)
        end
    elseif tA == 'T' && tB == 'T'
        @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
            νmin = max(1,k-Al,j-Bu)
            νmax = min(An,k+Au,j+Bl)

            tmp = zero(T)
            for ν=νmin:νmax
                tmp = tmp + inbands_getindex(A,ν,k) * inbands_getindex(B,j,ν)
            end
            inbands_setindex!(C,tmp,k,j)
        end
    end
    C
end

# use BLAS routine for positively banded BLASBandedMatrices
function _positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)
    # _banded_generic_matmatmul! is faster for sparse matrix
    if tA != 'N' || tB != 'N' || (Al + Au < 100 && Bl + Bu < 100)
        _banded_generic_matmatmul!(C, tA, tB, A, B)
    else
        # TODO: implement gbmm! routines for other flags
        gbmm!('N', 'N', one(T), A, B, zero(T), C)
    end
end


function generally_banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    if An != Bm || size(C, 1) != Am || size(C, 2) != Bn
        throw(DimensionMismatch("*"))
    end
    # TODO: checkbandmatch

    Al, Au = _bandwidths(tA, A)
    Bl, Bu = _bandwidths(tB, B)

    if (-Al > Au) || (-Bl > Bu)   # A or B has empty bands
        C[:,:] = zero(T)
    elseif Al < 0
        C[max(1,Bn+Al-1):Am, :] = zero(T)
        banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Al:An), _view(tB, B, 1-Al:An, :))
    elseif Au < 0
        C[1:-Au,:] = zero(T)
        banded_matmatmul!(view(C, 1-Au:Am,:), tA, tB, _view(tA, A, 1-Au:Am,:), B)
    elseif Bl < 0
        C[:, 1:-Bl] = zero(T)
        banded_matmatmul!(view(C, :, 1-Bl:Bn), tA, tB, A, _view(tB, B, :, 1-Bl:Bn))
    elseif Bu < 0
        C[:, max(1,Am+Bu-1):Bn] = zero(T)
        banded_matmatmul!(C, tA, tB, _view(tA, A, :, 1-Bu:Bm), _view(tB, B, 1-Bu:Bm, :))
    else
        positively_banded_matmatmul!(C, tA::Char, tB::Char, A, B)
    end
    C
end


banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V} = generally_banded_matmatmul!(C, tA, tB, A, B)
