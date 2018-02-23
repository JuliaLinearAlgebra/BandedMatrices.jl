
####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedLayout, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

abstract type MemoryLayout{T} end
struct UnknownLayout{T} <: MemoryLayout{T} end
abstract type AbstractStridedLayout{T} <: MemoryLayout{T} end
abstract type AbstractColumnMajor{T} <: AbstractStridedLayout{T} end
struct DenseColumnMajor{T} <: AbstractColumnMajor{T} end
struct ColumnMajor{T} <: AbstractColumnMajor{T} end
abstract type AbstractRowMajor{T} <: AbstractStridedLayout{T} end
struct DenseRowMajor{T} <: AbstractRowMajor{T} end
struct RowMajor{T} <: AbstractRowMajor{T} end
struct StridedLayout{T} <: AbstractStridedLayout{T} end


struct BandedLayout{T} <: MemoryLayout{T} end
struct SymBandedLayout{T} <: MemoryLayout{T} end

MemoryLayout(A::AbstractArray{T}) where T = UnknownLayout{T}()
MemoryLayout(A::Vector{T}) where T = DenseColumnMajor{T}()
MemoryLayout(A::Matrix{T}) where T = DenseColumnMajor{T}()

import Base: AbstractCartesianIndex, Slice, RangeIndex

MemoryLayout(A::SubArray) = submemorylayout(MemoryLayout(parent(A)), parentindexes(A))
submemorylayout(::MemoryLayout{T}, _) where T = UnknownLayout{T}()
submemorylayout(::AbstractColumnMajor{T}, ::Tuple{I}) where {T,I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor{T}()
submemorylayout(::AbstractStridedLayout{T}, ::Tuple{I}) where {T,I<:Union{RangeIndex,AbstractCartesianIndex}} =
    StridedLayout{T}()
submemorylayout(::AbstractColumnMajor{T}, ::Tuple{I,Int}) where {T,I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor{T}()
submemorylayout(::AbstractColumnMajor{T}, ::Tuple{I,Int}) where {T,I<:Slice} =
    DenseColumnMajor{T}()
submemorylayout(::AbstractRowMajor{T}, ::Tuple{Int,I}) where {T,I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}} =
    DenseColumnMajor{T}()
submemorylayout(::AbstractRowMajor{T}, ::Tuple{Int,I}) where {T,I<:Slice} =
    DenseColumnMajor{T}()
submemorylayout(::DenseColumnMajor{T}, ::Tuple{I1,I2}) where {T,I1<:Slice,I2<:AbstractUnitRange{Int}} =
    DenseColumnMajor{T}()
submemorylayout(::DenseColumnMajor{T}, ::Tuple{I1,I2}) where {T,I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    ColumnMajor{T}()
submemorylayout(::AbstractColumnMajor{T}, ::Tuple{I1,I2}) where {T,I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    ColumnMajor{T}()
submemorylayout(::AbstractRowMajor{T}, ::Tuple{I1,I2}) where {T,I1<:AbstractUnitRange{Int},I2<:Slice} =
    DenseRowMajor{T}()
submemorylayout(::AbstractRowMajor{T}, ::Tuple{I1,I2}) where {T,I1<:AbstractUnitRange{Int},I2<:AbstractUnitRange{Int}} =
    RowMajor{T}()
submemorylayout(::AbstractStridedLayout{T}, ::Tuple{I1,I2}) where {T,I1<:Union{RangeIndex,AbstractCartesianIndex},I2<:Union{RangeIndex,AbstractCartesianIndex}} =
    StridedLayout{T}()



# copy!

function banded_copy!(dest::AbstractMatrix{T}, src::AbstractMatrix) where T
    m,n = size(dest)
    (m,n) == size(src) || throw(DimensionMismatch())

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)
    (d_l ≥ s_l && d_u ≥ s_u) || throw(BandError(dest))
    for j=1:n
        for k = max(1,j-d_u):min(j-s_u-1,m)
            inbands_setindex!(dest, zero(T), k, j)
        end
        for k = max(1,j-s_u):min(j+s_l,m)
            inbands_setindex!(dest, inbands_getindex(src, k, j), k, j)
        end
        for k = max(1,j+s_l+1):min(j+d_l,m)
            inbands_setindex!(dest, zero(T), k, j)
        end
    end
    dest
end

# these are the routines of the banded interface of other AbstractMatrices
banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix) = _banded_axpy!(a, X, Y, MemoryLayout(X), MemoryLayout(Y))
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, ::BandedLayout, ::BandedLayout) =
    banded_generic_axpy!(a, X, Y)
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, notbandedX, notbandedY) =
    banded_dense_axpy!(a, X, Y)


# matrix * vector
banded_matvecmul!(c::AbstractVector, tA::Char, A::AbstractMatrix, b::AbstractVector) =
    _banded_matvecmul!(c, tA, A, b, MemoryLayout(c), MemoryLayout(A), MemoryLayout(b))
_banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{T}, b::AbstractVector{T},
                   ::StridedLayout{T}, ::BandedLayout{T}, ::StridedLayout{T}) where {T <: BlasFloat} =
    generally_banded_matvecmul!(c, tA, A, b)
_banded_matvecmul!(c::AbstractVector, tA::Char, A::AbstractMatrix, b::AbstractVector,
                   notblasc, notblasA, notblasb) =
    banded_generic_matvecmul!(c, tA, A, b)


banded_A_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'N', A, b)
banded_Ac_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'C', A, b)
banded_At_mul_B!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector) = banded_matvecmul!(c, 'T', A, b)


# matrix * matrix
banded_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix) =
    banded_matmatmul!(C, tA, tB, A, B, MemoryLayout(A), MemoryLayout(B))
banded_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix,
                  notblasA, notblasB) =
    banded_generic_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix,
                  ::BandedLayout, ::BandedLayout) =
    generally_banded_matmatmul!(C, tA, tB, A, B)



banded_A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'N', A, B)
banded_Ac_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'N', A, B)
banded_At_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'N', A, B)
banded_A_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'C', A, B)
banded_A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'N', 'T', A, B)
banded_Ac_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'C', 'C', A, B)
banded_At_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = banded_matmatmul!(C, 'T', 'T', A, B)



# some default functions



# additions and subtractions


@propagate_inbounds function banded_generic_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    n,m = size(X)
    if (n,m) ≠ size(Y)
        throw(BoundsError())
    end
    Xl, Xu = bandwidths(X)
    Yl, Yu = bandwidths(Y)

    @boundscheck if Xl > Yl
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j+Yl+1):min(j+Xl,n)
            if inbands_getindex(X, k, j) ≠ 0
                throw(BandError(X, (k,j)))
            end
        end
    end
    @boundscheck if Xu > Yu
        # test that all entries are zero in extra bands
        for j=1:size(X,2),k=max(1,j-Xu):min(j-Yu-1,n)
            if inbands_getindex(X, k, j) ≠ 0
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


positively_banded_matvecmul!(c::AbstractVector, tA::Char, A::AbstractMatrix, b::AbstractVector) =
    _positively_banded_matvecmul!(c, tA, A, b, MemoryLayout(c), MemoryLayout(A), MemoryLayout(b))
_positively_banded_matvecmul!(c::AbstractVector, tA::Char, A::AbstractMatrix, b::AbstractVector,
                              notblasc, notblasA, notblasb) =
    _banded_generic_matvecmul!(c, tA, A, b)

# use BLAS routine for positively banded BlasBanded
_positively_banded_matvecmul!(c::AbstractVector{T}, tA::Char, A::AbstractMatrix{T}, b::AbstractVector{T},
                                ::StridedLayout{T}, ::BandedLayout{T}, ::StridedLayout{T}) where {T <: BlasFloat} =
    gbmv!(tA, one(T), A, b, zero(T), c)

positively_banded_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix) =
    _positively_banded_matmatmul!(C, tA, tB, A, B, MemoryLayout(C), MemoryLayout(A), MemoryLayout(B))
_positively_banded_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix,
                              notblasC, notblasA, notblasb) =
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

banded_generic_matvecmul!(c::AbstractVector, tA::Char, A::AbstractMatrix, b::AbstractVector) =
    generally_banded_matvecmul!(c, tA, A, b)



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

# use BLAS routine for positively banded BlasBandedMatrices
function _positively_banded_matmatmul!(C::AbstractMatrix{T}, tA::Char, tB::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T},
                                       ::BandedLayout{T}, ::BandedLayout{T}, ::BandedLayout{T}) where {T}
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


banded_generic_matmatmul!(C::AbstractMatrix, tA::Char, tB::Char, A::AbstractMatrix, B::AbstractMatrix) =
    generally_banded_matmatmul!(C, tA, tB, A, B)



# add banded linear algebra routines between Typ1 and Typ2 both implementing
# the BandedMatrix interface
if VERSION < v"0.7-"
    macro _banded_banded_linalg(Typ1, Typ2)
        ret = quote
            Base.copy!(dest::$Typ1, src::$Typ2) = BandedMatrices.banded_copy!(dest,src)
            Base.BLAS.axpy!(a::Number, X::$Typ1, Y::$Typ2) = BandedMatrices.banded_axpy!(a, X, Y)

            function Base.:+(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
                n, m = size(A)
                ret = BandedMatrices.BandedMatrix(BandedMatrices.Zeros{promote_type(T,V)}(n, m), BandedMatrices.sumbandwidths(A, B))
                axpy!(one(T), A, ret)
                axpy!(one(V), B, ret)
                ret
            end

            function Base.:-(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
                n, m=size(A)
                ret = BandedMatrices.BandedMatrix(BandedMatrices.Zeros{promote_type(T,V)}(n, m), BandedMatrices.sumbandwidths(A, B))
                axpy!(one(T),  A, ret)
                axpy!(-one(V), B, ret)
                ret
            end


            Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = banded_matmatmul!(C, 'N', 'N', A, B)

            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

            Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            function Base.:*(A::$Typ1{T}, B::$Typ2{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                Y = BandedMatrix{promote_type(T,V)}(uninitialized, n, m, prodbandwidths(A, B)...)
                A_mul_B!(Y, A, B)
            end

            Base.LinAlg.Ac_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_B!(BandedMatrices.banded_similar('C', 'N', A, B, promote_type(T, V)), A, B)
            Base.LinAlg.A_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bc!(BandedMatrices.banded_similar('N', 'C', A, B, promote_type(T, V)), A, B)
            Base.LinAlg.Ac_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_Bc!(BandedMatrices.banded_similar('C', 'C', A, B, promote_type(T, V)), A, B)

            Base.LinAlg.At_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'N', A, B, promote_type(T, V)), A, B)
            Base.LinAlg.A_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bt!(BandedMatrices.banded_similar('N', 'T', A, B, promote_type(T, V)), A, B)
            Base.LinAlg.At_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'T', A, B, promote_type(T, V)), A, B)
        end
        esc(ret)
    end
else
    macro _banded_banded_linalg(Typ1, Typ2)
        ret = quote
            Base.copyto!(dest::$Typ1, src::$Typ2) = BandedMatrices.banded_copy!(dest,src)
            LinearAlgebra.BLAS.axpy!(a::Number, X::$Typ1, Y::$Typ2) = BandedMatrices.banded_axpy!(a, X, Y)

            function Base.:+(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
                n, m = size(A)
                ret = BandedMatrices.BandedMatrix(BandedMatrices.Zeros{promote_type(T,V)}(n, m), BandedMatrices.sumbandwidths(A, B))
                axpy!(one(T), A, ret)
                axpy!(one(V), B, ret)
                ret
            end

            function Base.:-(A::$Typ1{T}, B::$Typ2{V}) where {T,V}
                n, m=size(A)
                ret = BandedMatrices.BandedMatrix(BandedMatrices.Zeros{promote_type(T,V)}(n, m), BandedMatrices.sumbandwidths(A, B))
                axpy!(one(T),  A, ret)
                axpy!(-one(V), B, ret)
                ret
            end


            LinearAlgebra.A_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = banded_matmatmul!(C, 'N', 'N', A, B)

            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

            LinearAlgebra.At_mul_B!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.A_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.At_mul_Bt!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ1{U}, B::$Typ2{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            function Base.:*(A::$Typ1{T}, B::$Typ2{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                Y = BandedMatrix{promote_type(T,V)}(uninitialized, n, m, prodbandwidths(A, B)...)
                A_mul_B!(Y, A, B)
            end

            LinearAlgebra.Ac_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_B!(BandedMatrices.banded_similar('C', 'N', A, B, promote_type(T, V)), A, B)
            LinearAlgebra.A_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bc!(BandedMatrices.banded_similar('N', 'C', A, B, promote_type(T, V)), A, B)
            LinearAlgebra.Ac_mul_Bc(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = Ac_mul_Bc!(BandedMatrices.banded_similar('C', 'C', A, B, promote_type(T, V)), A, B)

            LinearAlgebra.At_mul_B(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'N', A, B, promote_type(T, V)), A, B)
            LinearAlgebra.A_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = A_mul_Bt!(BandedMatrices.banded_similar('N', 'T', A, B, promote_type(T, V)), A, B)
            LinearAlgebra.At_mul_Bt(A::$Typ1{T}, B::$Typ2{V}) where {T, V} = At_mul_B!(BandedMatrices.banded_similar('T', 'T', A, B, promote_type(T, V)), A, B)
        end
        esc(ret)
    end
end

macro banded_banded_linalg(Typ1, Typ2)
    ret = quote
        BandedMatrices.@_banded_banded_linalg($Typ1, $Typ2)
    end
    if Typ1 ≠ Typ2
        ret = quote
            $ret
            BandedMatrices.@_banded_banded_linalg($Typ2, $Typ1)
        end
    end
    esc(ret)
end

# add banded linear algebra routines for Typ implementing the BandedMatrix interface
if VERSION < v"0.7-"
    macro _banded_linalg(Typ)
        ret = quote
            BandedMatrices.@banded_banded_linalg($Typ, $Typ)

            Base.BLAS.axpy!(a::Number, X::$Typ, Y::AbstractMatrix) =
                    BandedMatrices.banded_axpy!(a, X, Y)
            function Base.:+(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
                ret = deepcopy(B)
                axpy!(one(T), A, ret)
                ret
            end
            function Base.:+(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
                n, m=size(A)
                ret = zeros(promote_type(T,V),n,m)
                axpy!(one(T), A,ret)
                axpy!(one(V), B,ret)
                ret
            end
            Base.:+(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = B + A

            function Base.:-(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
                ret = deepcopy(B)
                Base.scale!(ret, -one(T))
                Base.BLAS.axpy!(one(T), A, ret)
                ret
            end
            function Base.:-(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
                n, m= size(A)
                ret = zeros(promote_type(T,V),n,m)
                Base.BLAS.axpy!( one(T), A, ret)
                Base.BLAS.axpy!(-one(V), B, ret)
                ret
            end
            Base.:-(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = Base.scale!(B - A, -1)



            ## UniformScaling

            function Base.BLAS.axpy!(a::Number, X::UniformScaling, Y::$Typ{T}) where {T}
                BandedMatrices.checksquare(Y)
                α = a * X.λ
                @inbounds for k = 1:size(Y,1)
                    BandedMatrices.inbands_setindex!(Y, BandedMatrices.inbands_getindex(Y, k, k) + α, k, k)
                end
                Y
            end

            function Base.:+(A::$Typ, B::UniformScaling)
                ret = deepcopy(A)
                Base.BLAS.axpy!(1, B,ret)
            end

            Base.:+(A::UniformScaling, B::$Typ) = B + A

            function Base.:-(A::$Typ, B::UniformScaling)
                ret = deepcopy(A)
                axpy!(-1, B, ret)
            end

            function Base.:-(A::UniformScaling, B::$Typ)
                ret = deepcopy(B)
                Base.scale!(ret, -1)
                axpy!(1, A, ret)
            end

            Base.LinAlg.A_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'N', A, b)
            Base.LinAlg.Ac_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'C', A, b)
            Base.LinAlg.Ac_mul_B!(c::AbstractVector, A::$Typ{<:Real}, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'T', A, b)
            Base.LinAlg.At_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'T', A, b)

            Base.:*(A::$Typ{U}, b::StridedVector{V}) where {U, V} =
                Base.LinAlg.A_mul_B!(Vector{promote_type(U, V)}(uninitialized, size(A, 1)), A, b)


            Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)
            Base.LinAlg.A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)



            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

            Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.At_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
            Base.LinAlg.At_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.Ac_mul_B!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.A_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
            Base.LinAlg.Ac_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)


            function Base.:*(A::$Typ{T}, B::StridedMatrix{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
            end

            function Base.:*(A::StridedMatrix{T}, B::$Typ{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                A_mul_B!(Matrix{promote_type(T,V)}(n, m), A, B)
            end

            Base.LinAlg.Ac_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            Base.LinAlg.Ac_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            Base.LinAlg.A_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            Base.LinAlg.A_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            Base.LinAlg.Ac_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
            Base.LinAlg.Ac_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)

            Base.LinAlg.At_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            Base.LinAlg.At_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            Base.LinAlg.A_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            Base.LinAlg.A_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            Base.LinAlg.At_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
            Base.LinAlg.At_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
        end
        esc(ret)
    end
else
    macro _banded_linalg(Typ)
        ret = quote
            BandedMatrices.@banded_banded_linalg($Typ, $Typ)

            LinearAlgebra.BLAS.axpy!(a::Number, X::$Typ, Y::AbstractMatrix) =
                    BandedMatrices.banded_axpy!(a, X, Y)
            function Base.:+(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
                ret = deepcopy(B)
                axpy!(one(T), A, ret)
                ret
            end
            function Base.:+(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
                n, m=size(A)
                ret = zeros(promote_type(T,V),n,m)
                axpy!(one(T), A,ret)
                axpy!(one(V), B,ret)
                ret
            end
            Base.:+(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = B + A

            function Base.:-(A::$Typ{T}, B::AbstractMatrix{T}) where {T}
                ret = deepcopy(B)
                Base.scale!(ret, -one(T))
                LinearAlgebra.BLAS.axpy!(one(T), A, ret)
                ret
            end
            function Base.:-(A::$Typ{T}, B::AbstractMatrix{V}) where {T,V}
                n, m= size(A)
                ret = zeros(promote_type(T,V),n,m)
                LinearAlgebra.BLAS.axpy!( one(T), A, ret)
                LinearAlgebra.BLAS.axpy!(-one(V), B, ret)
                ret
            end
            Base.:-(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = Base.scale!(B - A, -1)



            ## UniformScaling

            function LinearAlgebra.BLAS.axpy!(a::Number, X::UniformScaling, Y::$Typ{T}) where {T}
                BandedMatrices.checksquare(Y)
                α = a * X.λ
                @inbounds for k = 1:size(Y,1)
                    BandedMatrices.inbands_setindex!(Y, BandedMatrices.inbands_getindex(Y, k, k) + α, k, k)
                end
                Y
            end

            function Base.:+(A::$Typ, B::UniformScaling)
                ret = deepcopy(A)
                LinearAlgebra.BLAS.axpy!(1, B,ret)
            end

            Base.:+(A::UniformScaling, B::$Typ) = B + A

            function Base.:-(A::$Typ, B::UniformScaling)
                ret = deepcopy(A)
                axpy!(-1, B, ret)
            end

            function Base.:-(A::UniformScaling, B::$Typ)
                ret = deepcopy(B)
                Base.scale!(ret, -1)
                axpy!(1, A, ret)
            end

            LinearAlgebra.A_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'N', A, b)
            LinearAlgebra.Ac_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'C', A, b)
            LinearAlgebra.Ac_mul_B!(c::AbstractVector, A::$Typ{<:Real}, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'T', A, b)
            LinearAlgebra.At_mul_B!(c::AbstractVector, A::$Typ, b::AbstractVector) =
                BandedMatrices.banded_matvecmul!(c, 'T', A, b)

            Base.:*(A::$Typ{U}, b::StridedVector{V}) where {U, V} =
                LinearAlgebra.A_mul_B!(Vector{promote_type(U, V)}(uninitialized,size(A, 1)), A, b)


            LinearAlgebra.A_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)
            LinearAlgebra.A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'N', A, B)



            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'N', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'C', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'C', 'C', A, B)

            LinearAlgebra.At_mul_B!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.At_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.A_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.A_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.At_mul_Bt!(C::AbstractMatrix, A::$Typ, B::AbstractMatrix) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
            LinearAlgebra.At_mul_Bt!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)

            # override the Ac_mul_B, A_mul_Bc and Ac_mul_c for real values
            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.Ac_mul_B!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, 'T', 'N', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.A_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'N', 'T', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix{T}, A::$Typ{U}, B::AbstractMatrix{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)
            LinearAlgebra.Ac_mul_Bc!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, B::$Typ{V}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, 'T', 'T', A, B)


            function Base.:*(A::$Typ{T}, B::StridedMatrix{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                A_mul_B!(Matrix{promote_type(T,V)}(uninitialized,n,m), A, B)
            end

            function Base.:*(A::StridedMatrix{T}, B::$Typ{V}) where {T, V}
                n, m = size(A,1), size(B,2)
                A_mul_B!(Matrix{promote_type(T,V)}(uninitialized,n,m), A, B)
            end

            LinearAlgebra.Ac_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            LinearAlgebra.Ac_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            LinearAlgebra.A_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            LinearAlgebra.A_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            LinearAlgebra.Ac_mul_Bc(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
            LinearAlgebra.Ac_mul_Bc(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = Ac_mul_Bc!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)

            LinearAlgebra.At_mul_B(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            LinearAlgebra.At_mul_B(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 2)), A, B)
            LinearAlgebra.A_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            LinearAlgebra.A_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = A_mul_Bt!(Matrix{promote_type(T,V)}(size(A, 1), size(B, 1)), A, B)
            LinearAlgebra.At_mul_Bt(A::$Typ{T}, B::StridedMatrix{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
            LinearAlgebra.At_mul_Bt(A::StridedMatrix{T}, B::$Typ{V}) where {T, V} = At_mul_B!(Matrix{promote_type(T,V)}(size(A, 2), size(B, 1)), A, B)
        end
        esc(ret)
    end
end

macro banded_linalg(Typ)
    ret = quote
        BandedMatrices.@_banded_linalg($Typ)
        BandedMatrices.@banded_banded_linalg($Typ, BandedMatrices.AbstractBandedMatrix)
    end
    esc(ret)
end

# add routines for banded interface
macro banded_interface(Typ)
    ret = quote
        BandedMatrices.MemoryLayout(A::$Typ) = BandedMatrices.BandedLayout{eltype(A)}()
        BandedMatrices.isbanded(::$Typ) = true
    end
    esc(ret)
end

macro banded(Typ)
    ret = quote
        BandedMatrices.@banded_interface($Typ)
        BandedMatrices.@banded_linalg($Typ)
    end
    esc(ret)
end
