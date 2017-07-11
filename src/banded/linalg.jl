# additions and subtractions

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


function banded_axpy!{T}(a::Number,X,S::BandedSubBandedMatrix{T})
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

Base.BLAS.axpy!{T}(a::Number,X::BandedMatrix,S::BandedSubBandedMatrix{T}) =
    banded_axpy!(a,X,S)

function Base.BLAS.axpy!{T1,T2}(a::Number,X::BandedSubBandedMatrix{T1},Y::BandedSubBandedMatrix{T2})
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

function Base.BLAS.axpy!{T}(a::Number,X::BandedSubBandedMatrix{T},Y::BandedMatrix)
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

function +{T,V}(A::BandedMatrix{T},B::BandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(1.,B,ret)

    ret
end

function -{T,V}(A::BandedMatrix{T}, B::BandedMatrix{V})
    if size(A) != size(B)
        throw(DimensionMismatch("+"))
    end
    n,m=size(A,1),size(A,2)

    ret = bzeros(promote_type(T,V),n,m,max(A.l,B.l),max(A.u,B.u))
    BLAS.axpy!(1.,A,ret)
    BLAS.axpy!(-1.,B,ret)

    ret
end

# matrix * vector

function banded_generic_matvecmul!{T, U, V}(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V})
    @inbounds c[:] = zero(T)
    @inbounds for j = 1:size(A,2), k = colrange(A,j)
        c[k] += inbands_getindex(A,k,j)*b[j]
    end
    c
end

_banded_matvecmul!{T <: BlasFloat}(c::StridedVector{T}, A::BLASBandedMatrix{T}, b::StridedVector{T}) = gbmv!('N',one(T),A,b,zero(T),c)
_banded_matvecmul!{T, U, V}(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) = banded_generic_matvecmul!(c, A, b)

function banded_matvecmul!{T, U, V}(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V})
    m,n = size(A)

    @boundscheck if length(c) ≠ m || length(b) ≠ n
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
        _banded_matvecmul!(c, A, b)
    end
    c
end

A_mul_B!{T, U, V}(c::AbstractVector{T}, A::BLASBandedMatrix{U}, b::AbstractVector{V}) = banded_matvecmul!(c, A, b)
*{U, V}(A::BLASBandedMatrix{U},b::StridedVector{V}) = A_mul_B!(Vector{promote_type(U, V)}(size(A, 1)), A, b)


function banded_generic_matmatmul!{T, U, V}(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::AbstractMatrix{V})
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
    Cl,Cu = Al+Bl, Au+Bu

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

_banded_matmatmul!{T <: BlasFloat}(C::BLASBandedMatrix{T} ,A::BLASBandedMatrix{T}, B::BLASBandedMatrix{T}) = gbmm!(one(T), A, B, zero(T), C)
_banded_matmatmul!{T <: BlasFloat}(C::StridedMatrix{T} ,A::BLASBandedMatrix{T}, B::StridedMatrix{T}) = gbmm!(one(T), A, B, zero(T), C)
_banded_matmatmul!{T <: BlasFloat}(C::StridedMatrix{T} ,A::StridedMatrix{T}, B::BLASBandedMatrix{T}) = gbmm!(one(T), A, B, zero(T), C)
_banded_matmatmul!{T, U, V}(C::AbstractMatrix{T} ,A::AbstractMatrix{U}, B::AbstractMatrix{V}) = banded_generic_matmatmul!(C, A, B)

function banded_matmatmul!{T, U, V}(C::AbstractMatrix{T} ,A::AbstractMatrix{U}, B::AbstractMatrix{V})
    Am, An = size(A)
    Bm, Bn = size(B)
    @boundscheck if size(A,2) != size(B,1) || size(C,1) != Am || size(C,2) != Bn
        throw(DimensionMismatch("*"))
    end
    # TODO: checkbandmatch

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
    elseif Al + Au < 100 && Bl + Bu < 100
        # for narrow matrices, `banded_generic_matmatmul` is faster
        banded_generic_matmatmul!(C, A, B)
    else
        _banded_matmatmul!(C, A, B)
    end
    C
end

A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, A, B)
A_mul_B!(C::AbstractMatrix ,A::BLASBandedMatrix, B::StridedMatrix) = banded_matmatmul!(C, A, B)
A_mul_B!(C::AbstractMatrix ,A::StridedMatrix, B::BLASBandedMatrix) = banded_matmatmul!(C, A, B)


function *{T, V}(A::BLASBandedMatrix{T},B::BLASBandedMatrix{V})
    Al, Au = bandwidths(A)
    Bl, Bu = bandwidths(B)
    n,m = size(A,1),size(B,2)
    Y = BandedMatrix(promote_type(T,V),n,m,Al+Bl,Au+Bu)
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
