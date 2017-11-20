# additions and subtractions

function banded_generic_axpy!(a::Number, X::AbstractMatrix{U}, Y::AbstractMatrix{V}) where {U, V}
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

function banded_dense_axpy!(a::Number, X::AbstractMatrix ,Y::AbstractMatrix)
    if size(X) != size(Y)
        throw(DimensionMismatch("+"))
    end
    @inbounds for j=1:size(X,2),k=colrange(X,j)
        Y[k,j]+=a*inbands_getindex(X,k,j)
    end
    Y
end

banded_axpy!(a::Number, X::BLASBandedMatrix ,Y::BLASBandedMatrix) = banded_generic_axpy!(a, X, Y)
banded_axpy!(a::Number, X::BLASBandedMatrix ,Y::AbstractMatrix) = banded_dense_axpy!(a, X, Y)

axpy!(a::Number, X::BLASBandedMatrix, Y::BLASBandedMatrix) = banded_axpy!(a, X, Y)
axpy!(a::Number, X::BLASBandedMatrix, Y::AbstractMatrix) = banded_axpy!(a, X, Y)


function +(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T,V}
    n, m=size(A)
    ret = BandedMatrix(Zeros{promote_type(T,V)}(n,m), sumbandwidths(A, B))
    axpy!(1.,A,ret)
    axpy!(1.,B,ret)
    ret
end

function +(A::BLASBandedMatrix{T},B::AbstractMatrix{T}) where {T}
    ret = deepcopy(B)
    axpy!(one(T),A,ret)
    ret
end

function +(A::BLASBandedMatrix{T},B::AbstractMatrix{V}) where {T,V}
    n, m=size(A)
    ret = zeros(promote_type(T,V),n,m)
    axpy!(one(T),A,ret)
    axpy!(one(V),B,ret)
    ret
end

+(A::AbstractMatrix{T},B::BLASBandedMatrix{V}) where {T,V} = B+A


function -(A::BLASBandedMatrix{T}, B::BLASBandedMatrix{V}) where {T,V}
    n, m=size(A)
    ret = BandedMatrix(Zeros{promote_type(T,V)}(n,m), sumbandwidths(A, B))
    axpy!(one(T),A,ret)
    axpy!(-one(V),B,ret)
    ret
end

function -(A::BLASBandedMatrix{T},B::AbstractMatrix{T}) where {T}
    ret = deepcopy(B)
    Base.scale!(ret,-1)
    axpy!(one(T),A,ret)
    ret
end


function -(A::BLASBandedMatrix{T},B::AbstractMatrix{V}) where {T,V}
    n, m=size(A)
    ret = zeros(promote_type(T,V),n,m)
    axpy!(one(T),A,ret)
    axpy!(-one(V),B,ret)
    ret
end

-(A::AbstractMatrix{T},B::BLASBandedMatrix{V}) where {T,V} = Base.scale!(B-A,-1)



## UniformScaling

function axpy!(a::Number,X::UniformScaling,Y::BLASBandedMatrix{T}) where {T}
    checksquare(Y)
    α = a * X.λ
    @inbounds for k = 1:size(Y,1)
        inbands_setindex!(Y, inbands_getindex(Y, k, k) + α, k, k)
    end
    Y
end

function +(A::BLASBandedMatrix, B::UniformScaling)
    ret = deepcopy(A)
    axpy!(1,B,ret)
end

+(A::UniformScaling, B::BLASBandedMatrix) = B+A

function -(A::BLASBandedMatrix, B::UniformScaling)
    ret = deepcopy(A)
    axpy!(-1,B,ret)
end

function -(A::UniformScaling, B::BLASBandedMatrix)
    ret = deepcopy(B)
    Base.scale!(ret,-1)
    axpy!(1,A,ret)
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

positively_banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V} = _banded_generic_matvecmul!(c, tA, A, b)
# use BLAS routine for positively banded BLASBandedMatrix
positively_banded_matvecmul!(c::StridedVector{T}, tA::Char, A::BLASBandedMatrix{T}, b::StridedVector{T}) where {T <: BlasFloat} = gbmv!(tA, one(T), A, b, zero(T), c)

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

banded_matvecmul!(c::StridedVector{T}, tA::Char, A::BLASBandedMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat} = generally_banded_matvecmul!(c, tA, A, b)

banded_generic_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) where {T, U, V} = generally_banded_matvecmul!(c, tA, A, b)

A_mul_B!(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) where {T, U, V} = banded_matvecmul!(c, 'N', A, b)
Ac_mul_B!(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) where {T, U, V} = banded_matvecmul!(c, 'C', A, b)
Ac_mul_B!(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) where {T, U<:Real, V} = banded_matvecmul!(c, 'T', A, b)
At_mul_B!(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) where {T, U, V} = banded_matvecmul!(c, 'T', A, b)

*(A::BLASBandedMatrix{U}, b::StridedVector{V}) where {U, V} =
    A_mul_B!(Vector{promote_type(U, V)}(size(A, 1)), A, b)


# matrix * matrix
function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
    Am, An = _size(tA, A)
    Bm, Bn = _size(tB, B)
    Al,Au = _bandwidths(tA, A)
    Bl,Bu = _bandwidths(tB, B)
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
    Al,Au = _bandwidths(tA, A)
    Bl,Bu = _bandwidths(tB, B)
    # _banded_generic_matmatmul! is faster for sparse matrix
    if tA != 'N' || tB != 'N' || (Al + Au < 100 && Bl + Bu < 100)
        _banded_generic_matmatmul!(C, tA, tB, A, B)
    else
        # TODO: implement gbmm! routines for other flags
        gbmm!('N', 'N', one(T), A, B, zero(T), C)
    end
end

positively_banded_matmatmul!(C::BLASBandedMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::BLASBandedMatrix{T}) where {T <: BlasFloat} = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!(C::StridedMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::StridedMatrix{T}) where {T <: BlasFloat} = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!(C::StridedMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::BLASBandedMatrix{T}) where {T <: BlasFloat} = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V} = _banded_generic_matmatmul!(C, tA, tB, A, B)


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


banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::BLASBandedMatrix{T}) where {T <: BlasFloat} = generally_banded_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::StridedMatrix{T}) where {T <: BlasFloat} = generally_banded_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::BLASBandedMatrix{T}) where {T <: BlasFloat} = generally_banded_matmatmul!(C, tA, tB, A, B)

banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V} = generally_banded_matmatmul!(C, tA, tB, A, B)

A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
A_mul_B!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)

Ac_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'C', 'N', A, B)
Ac_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'N', A, B)
Ac_mul_B!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'C', 'N', A, B)
A_mul_Bc!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'C', A, B)
A_mul_Bc!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'C', A, B)
A_mul_Bc!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'C', A, B)
Ac_mul_Bc!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'C', 'C', A, B)
Ac_mul_Bc!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'C', A, B)
Ac_mul_Bc!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'C', 'C', A, B)

At_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'T', 'N', A, B)
At_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'N', A, B)
At_mul_B!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'T', 'N', A, B)
A_mul_Bt!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'T', A, B)
A_mul_Bt!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'T', A, B)
A_mul_Bt!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'T', A, B)
At_mul_Bt!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'T', 'T', A, B)
At_mul_Bt!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'T', A, B)
At_mul_Bt!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'T', 'T', A, B)

# override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
Ac_mul_B!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::BLASBandedMatrix{V}) where {T, U<:Real, V} = banded_matmatmul!(C, 'T', 'N', A, B)
Ac_mul_B!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::AbstractMatrix{V}) where {T, U<:Real, V} = banded_matmatmul!(C, 'T', 'N', A, B)
Ac_mul_B!(C::AbstractMatrix{T} ,A::AbstractMatrix{U}, B::BLASBandedMatrix{V}) where {T, U<:Real, V} = banded_matmatmul!(C, 'T', 'N', A, B)
A_mul_Bc!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::BLASBandedMatrix{V}) where {T, U, V<:Real} = banded_matmatmul!(C, 'N', 'T', A, B)
A_mul_Bc!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::AbstractMatrix{V}) where {T, U, V<:Real} = banded_matmatmul!(C, 'N', 'T', A, B)
A_mul_Bc!(C::AbstractMatrix{T} ,A::AbstractMatrix{U}, B::BLASBandedMatrix{V}) where {T, U, V<:Real} = banded_matmatmul!(C, 'N', 'T', A, B)
Ac_mul_Bc!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::BLASBandedMatrix{V}) where {T, U<:Real, V<:Real} = banded_matmatmul!(C, 'T', 'T', A, B)
Ac_mul_Bc!(C::AbstractMatrix{T} ,A::BLASBandedMatrix{U}, B::AbstractMatrix{V}) where {T, U<:Real, V<:Real} = banded_matmatmul!(C, 'T', 'T', A, B)
Ac_mul_Bc!(C::AbstractMatrix{T} ,A::AbstractMatrix{U}, B::BLASBandedMatrix{V}) where {T, U<:Real, V<:Real} = banded_matmatmul!(C, 'T', 'T', A, B)



function *(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V}
    n, m = size(A,1), size(B,2)
    Y = BandedMatrix{promote_type(T,V)}(n, m, prodbandwidths(A, B)...)
    A_mul_B!(Y,A,B)
end

function *(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V}
    n, m = size(A,1), size(B,2)
    A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
end

function *(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V}
    n, m = size(A,1), size(B,2)
    A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
end


Ac_mul_B(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = Ac_mul_B!(banded_similar('C', 'N', A, B, promote_type(T, V)), A, B)
Ac_mul_B(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
Ac_mul_B(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
A_mul_Bc(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = A_mul_Bc!(banded_similar('N', 'C', A, B, promote_type(T, V)), A, B)
A_mul_Bc(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
A_mul_Bc(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
Ac_mul_Bc(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = Ac_mul_Bc!(banded_similar('C', 'C', A, B, promote_type(T, V)), A, B)
Ac_mul_Bc(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
Ac_mul_Bc(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)

At_mul_B(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = At_mul_B!(banded_similar('T', 'N', A, B, promote_type(T, V)), A, B)
At_mul_B(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
At_mul_B(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
A_mul_Bt(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = A_mul_Bt!(banded_similar('N', 'T', A, B, promote_type(T, V)), A, B)
A_mul_Bt(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
A_mul_Bt(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
At_mul_Bt(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = At_mul_B!(banded_similar('T', 'T', A, B, promote_type(T, V)), A, B)
At_mul_Bt(A::BLASBandedMatrix{T},B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
At_mul_Bt(A::StridedMatrix{T},B::BLASBandedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)


## Method definitions for generic eltypes - will make copies

# Direct and transposed algorithms
for typ in [BandedMatrix, BandedLU]
    for fun in [:A_ldiv_B!, :At_ldiv_B!]
        @eval function $fun(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
            checksquare(A)
            AA, BB = _convert_to_blas_type(A, B)
            $fun(lufact(AA), BB) # call BlasFloat versions
        end
    end
    # \ is different because it needs a copy, but we have to avoid ambiguity
    @eval function (\)(A::$typ{T}, B::VecOrMat{Complex{T}}) where {T<:BlasReal}
        checksquare(A)
        A_ldiv_B!(convert($typ{Complex{T}}, A), copy(B)) # goes to BlasFloat call
    end
    @eval function (\)(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Number, S<:Number}
        checksquare(A)
        TS = _promote_to_blas_type(T, S)
        A_ldiv_B!(convert($typ{TS}, A), copy_oftype(B, TS)) # goes to BlasFloat call
    end
end

# Hermitian conjugate
for typ in [BandedMatrix, BandedLU]
    @eval function Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Complex, S<:Number}
        checksquare(A)
        AA, BB = _convert_to_blas_type(A, B)
        Ac_ldiv_B!(lufact(AA), BB) # call BlasFloat versions
    end
    @eval Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Real, S<:Real} =
        At_ldiv_B!(A, B)
    @eval Ac_ldiv_B!(A::$typ{T}, B::StridedVecOrMat{S}) where {T<:Real, S<:Complex} =
        At_ldiv_B!(A, B)
end


# Method definitions for BlasFloat types - no copies

# Direct and transposed algorithms
for (ch, fname) in zip(('N', 'T'), (:A_ldiv_B!, :At_ldiv_B!))
    # provide A*_ldiv_B!(::BandedLU, ::StridedVecOrMat) for performance
    @eval function $fname(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        gbtrs!($ch, A.l, A.u, A.m, A.data, A.ipiv, B)
    end
    # provide A*_ldiv_B!(::BandedMatrix, ::StridedVecOrMat) for generality
    @eval function $fname(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasFloat}
        checksquare(A)
        $fname(lufact(A), B)
    end
end

# Hermitian conjugate algorithms - same two routines as above
function Ac_ldiv_B!(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
    checksquare(A)
    gbtrs!('C', A.l, A.u, A.m, A.data, A.ipiv, B)
end

function Ac_ldiv_B!(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasComplex}
    checksquare(A)
    Ac_ldiv_B!(lufact(A), B)
end

# fall back for real inputs
Ac_ldiv_B!(A::BandedLU{T}, B::StridedVecOrMat{T}) where {T<:BlasReal} =
    At_ldiv_B!(A, B)
Ac_ldiv_B!(A::BandedMatrix{T}, B::StridedVecOrMat{T}) where {T<:BlasReal} =
    At_ldiv_B!(A, B)
