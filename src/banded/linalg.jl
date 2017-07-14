# additions and subtractions

function banded_generic_axpy!{U, V}(a::Number, X::AbstractMatrix{U}, Y::AbstractMatrix{V})
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

function banded_dense_axpy!(a::Number, X::BLASBandedMatrix ,Y::AbstractMatrix)
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


function +{T,V}(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V})
    n, m=size(A)
    ret = bzeros(promote_type(T,V),n,m,sumbandwidths(A, B)...)
    axpy!(1.,A,ret)
    axpy!(1.,B,ret)
    ret
end

function +{T}(A::BLASBandedMatrix{T},B::AbstractMatrix{T})
    ret = deepcopy(B)
    axpy!(one(T),A,ret)
    ret
end

function +{T,V}(A::BLASBandedMatrix{T},B::AbstractMatrix{V})
    n, m=size(A)
    ret = zeros(promote_type(T,V),n,m)
    axpy!(one(T),A,ret)
    axpy!(one(V),B,ret)
    ret
end

+{T,V}(A::AbstractMatrix{T},B::BLASBandedMatrix{V}) = B+A


function -{T,V}(A::BLASBandedMatrix{T}, B::BLASBandedMatrix{V})
    n, m=size(A)
    ret = bzeros(promote_type(T,V),n,m,sumbandwidths(A, B)...)
    axpy!(one(T),A,ret)
    axpy!(-one(V),B,ret)
    ret
end

function -{T}(A::BLASBandedMatrix{T},B::AbstractMatrix{T})
    ret = deepcopy(B)
    Base.scale!(ret,-1)
    axpy!(one(T),A,ret)
    ret
end


function -{T,V}(A::BLASBandedMatrix{T},B::AbstractMatrix{V})
    n, m=size(A)
    ret = zeros(promote_type(T,V),n,m)
    axpy!(one(T),A,ret)
    axpy!(-one(V),B,ret)
    ret
end

-{T,V}(A::AbstractMatrix{T},B::BLASBandedMatrix{V}) = Base.scale!(B-A,-1)



## UniformScaling

function axpy!{T}(a::Number,X::UniformScaling,Y::BLASBandedMatrix{T})
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

function _banded_generic_matvecmul!{T, U, V}(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V})
    if tA ≠ 'N'
        error("Only 'N' flag is supported.")
    end
    @inbounds c[:] = zero(T)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[k] += inbands_getindex(A,k,j)*b[j]
    end
    c
end

positively_banded_matvecmul!{T, U, V}(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) = _banded_generic_matvecmul!(c, tA, A, b)
# use BLAS routine for positively banded BLASBandedMatrix
positively_banded_matvecmul!{T <: BlasFloat}(c::StridedVector{T}, tA::Char, A::BLASBandedMatrix{T}, b::StridedVector{T}) = gbmv!(tA, one(T), A, b, zero(T), c)

function generally_banded_matvecmul!{T, U, V}(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V})
    m,n = size(A)

    if length(c) ≠ m || length(b) ≠ n
        throw(DimensionMismatch("*"))
    end

    l,u = bandwidths(A)
    if -l > u
        # no bands
        c[:] = zero(T)
    elseif l < 0
        A_mul_B!(c,view(A,:,1-l:n),view(b,1-l:n))
    elseif u < 0
        c[1:-u] = zero(T)
        A_mul_B!(view(c,1-u:m),view(A,1-u:m,:),b)
    else
        positively_banded_matvecmul!(c, tA, A, b)
    end
    c
end

banded_matvecmul!{T <: BlasFloat}(c::StridedVector{T}, tA::Char, A::BLASBandedMatrix{T}, b::AbstractVector{T}) = generally_banded_matvecmul!(c, tA, A, b)

banded_generic_matvecmul!{T, U, V}(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{U}, b::AbstractVector{V}) = generally_banded_matvecmul!(c, tA, A, b)

A_mul_B!{T, U, V}(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) = banded_matvecmul!(c, 'N', A, b)

*{U, V}(A::BLASBandedMatrix{U}, b::StridedVector{V}) = A_mul_B!(Vector{promote_type(U, V)}(size(A, 1)), A, b)


# matrix * matrix
function _banded_generic_matmatmul!{T, U, V}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V})
    if tA ≠ 'N' || tB ≠ 'N'
        error("Only 'N' flag is supported.")
    end
    Am, An = size(A)
    Bm, Bn = size(B)
    if isbanded(A)
        Al,Au = bandwidths(A)
    else
        Al,Au = size(A,1)-1,size(A,2)-1
    end
    if isbanded(B)
        Bl,Bu = bandwidths(B)
    else
        Bl,Bu = size(B,1)-1,size(B,2)-1
    end
    Cl,Cu = prodbandwidths(A, B)

    @inbounds for j = 1:Bn, k = max(j-Cu, 1):max(min(j+Cl, Am), 0)
        νmin = max(1,k-Al,j-Bu)
        νmax = min(size(A,2),k+Au,j+Bl)

        tmp = zero(T)
        for ν=νmin:νmax
            tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,ν,j)
        end
        inbands_setindex!(C,tmp,k,j)
    end

    C
end

# use BLAS routine for positively banded BLASBandedMatrices
function _positively_banded_matmatmul!{T <: BlasFloat}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T})
    Al,Au = bandwidths(A)
    Bl,Bu = bandwidths(B)
    # _banded_generic_matmatmul! is faster for sparse matrix
    if Al + Au < 100 && Bl + Bu < 100
        _banded_generic_matmatmul!(C, tA, tB, A, B)
    else
        gbmm!(tA, tB, one(T), A, B, zero(T), C)
    end
end

positively_banded_matmatmul!{T <: BlasFloat}(C::BLASBandedMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::BLASBandedMatrix{T}) = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!{T <: BlasFloat}(C::StridedMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::StridedMatrix{T}) = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!{T <: BlasFloat}(C::StridedMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::BLASBandedMatrix{T}) = _positively_banded_matmatmul!(C, tA, tB, A, B)
positively_banded_matmatmul!{T, U, V}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) = _banded_generic_matmatmul!(C, tA, tB, A, B)


function generally_banded_matmatmul!{T, U, V}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V})
    Am, An = size(A)
    Bm, Bn = size(B)
    if size(A,2) != size(B,1) || size(C,1) != Am || size(C,2) != Bn
        throw(DimensionMismatch("*"))
    end
    # TODO: checkbandmatch

    Al,Au = bandwidths(A)
    Bl,Bu = bandwidths(B)

    if (-Al > Au) || (-Bl > Bu)   # A or B has empty bands
        C[:,:] = zero(T)
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
        positively_banded_matmatmul!(C, tA::Char, tB::Char, A, B)
    end
    C
end


banded_matmatmul!{T <: BlasFloat}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::BLASBandedMatrix{T}) = generally_banded_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!{T <: BlasFloat}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::BLASBandedMatrix{T}, B::StridedMatrix{T}) = generally_banded_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!{T <: BlasFloat}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::StridedMatrix{T}, B::BLASBandedMatrix{T}) = generally_banded_matmatmul!(C, tA, tB, A, B)

banded_generic_matmatmul!{T, U, V}(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{U}, B::AbstractMatrix{V}) = generally_banded_matmatmul!(C, tA, tB, A, B)

A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
A_mul_B!(C::AbstractMatrix ,A::AbstractMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)

function *{T, V}(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V})
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    n,m = size(A,1),size(B,2)
    Y = BandedMatrix(promote_type(T,V),n,m,prodbandwidths(A, B)...)
    A_mul_B!(Y,A,B)
end

function *{T, V}(A::BLASBandedMatrix{T},B::StridedMatrix{V})
    n,m=size(A,1),size(B,2)
    A_mul_B!(Matrix{promote_type(T,V)}(n,m),A,B)
end

function *{T, V}(A::StridedMatrix{T},B::BLASBandedMatrix{V})
    n,m=size(A,1),size(B,2)
    A_mul_B!(Matrix{promote_type(T,V)}(n,m),A,B)
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
