####
# Matrix memory layout traits
#
# if MemoryLayout(A) returns BandedColumnMajor, you must override
# pointer and leadingdimension
# in addition to the banded matrix interface
####

struct BandedLayout <: AbstractBandedLayout end
struct BandedColumns{ML} <: AbstractBandedLayout end
struct BandedRows{ML} <: AbstractBandedLayout end

const BandedColumnMajor = BandedColumns{<:AbstractColumnMajor}
const BandedRowMajor = BandedRows{<:AbstractColumnMajor}
BandedColumnMajor() = BandedColumns{DenseColumnMajor}()
BandedRowMajor() = BandedRows{DenseRowMajor}()

transposelayout(M::BandedColumns{ML}) where ML = BandedRows{ML}()
transposelayout(M::BandedRows{ML}) where ML = BandedColumns{ML}()
conjlayout(::Type{<:Complex}, ::M) where M<:AbstractBandedLayout = ConjLayout{M}()

# bandedrowsdata(Ac::Adjoint) = bandeddata(parent(Ac))'
# bandedrowsdata(Ac::Transpose) = transpose(bandeddata(parent(Ac)))
# bandedrowsdata(A) = bandedrowsdata(MemoryLayout(A), A)


# Here we override broadcasting for banded matrices.
# The design is to to exploit the broadcast machinery so that
# banded matrices that conform to the banded matrix interface but are not
# <: AbstractBandedMatrix can get access to fast copyto!, lmul!, rmul!, axpy!, etc.
# using broadcast variants (B .= A, B .= 2.0 .* A, etc.)



struct BandedStyle <: AbstractArrayStyle{2} end
BandedStyle(::Val{2}) = BandedStyle()
BroadcastStyle(::Type{<:AbstractBandedMatrix}) = BandedStyle()
BroadcastStyle(::Type{<:Adjoint{<:Any,Mat}}) where Mat<:AbstractBandedMatrix = BroadcastStyle(Mat)
BroadcastStyle(::Type{<:Transpose{<:Any,Mat}}) where Mat<:AbstractBandedMatrix = BroadcastStyle(Mat)
BroadcastStyle(::Type{<:SubArray{<:Any,2,<:AbstractBandedMatrix,<:NTuple{2,AbstractUnitRange{Int}}}}) = BandedStyle()
BroadcastStyle(::DefaultArrayStyle{2}, ::BandedStyle) = BandedStyle()
BroadcastStyle(::BandedStyle, ::DefaultArrayStyle{2}) = BandedStyle()


size(bc::Broadcasted{BandedStyle}) = length.(axes(bc))
isbanded(bc::Broadcasted{BandedStyle}) = true

####
# Default to standard Array broadcast
#
# This is because, for example, exp.(B) is not banded
####


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle}) =
   copyto!(dest, Broadcasted{DefaultArrayStyle{2}}(bc.f, bc.args, bc.axes))
##
# copyto!
##

_copyto!(dest_L::AbstractBandedLayout, src_L::AbstractBandedLayout, dest::AbstractMatrix, src::AbstractMatrix) =
    _banded_broadcast!(dest, identity, src, dest_L, src_L)


function checkbroadcastband(dest, sizesrc, bndssrc)
    size(dest) == sizesrc || throw(DimensionMismatch())
    min(sizesrc[1],bndssrc[1]+1) ≤ min(bandwidth(dest,1)+1,size(dest,1))  &&
        min(sizesrc[2],bndssrc[2]+1) ≤ min(bandwidth(dest,2)+2,size(dest,2)) || throw(BandError(dest,size(dest,2)-1))
end

###########
# matrix broadcast
###########

function checkzerobands(dest, f, A::AbstractMatrix)
    m,n = size(A)
    d_l, d_u = bandwidths(dest)
    l, u = bandwidths(A)

    if !(d_l >= l && d_u >= u)
        for j = rowsupport(A)
            for k = max(1,j-u) : min(j-d_u-1,m)
                iszero(f(A[k,j])) || throw(BandError(dest,j-k))
            end
            for k = max(1,j+d_l+1) : min(j+l,m)
                iszero(f(A[k,j])) || throw(BandError(dest,j-k))
            end
        end
    end
end

function _banded_broadcast!(dest::AbstractMatrix, f, src::AbstractMatrix{T}, _1, _2) where T
    z = f(zero(T))
    iszero(z) || checkbroadcastband(dest, size(src), bandwidths(broadcasted(f, src)))
    m,n = size(dest)
    m == n == 0 && return dest

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)

    checkzerobands(dest, f, src)

    _banded_broadcast_anylayout!(dest, src, f, z, (s_l, s_u), (d_l, d_u), m)

    dest
end

# columns to the left (start) and right (stop) of the non-zero columns _nzcols
@inline _startzcols((m,n), (l,u)) = 1:-l
@inline _nzcols((m,n), (l,u)) = max(1-l,1):min(m+u,n)
@inline _stopzcols((m,n), (l,u)) = m+u+1:n

# columns in the "start up", bulk and "stop
@inline _startcols((m,n), (l,u)) = max(1-l,1):min(u,n)
@inline _bulkcols((m,n), (l,u)) = max(u+1,1):min(m-l,n)
@inline _stopcols((m,n), (l,u)) = max(m-l+1,u+1,1):min(m+u,n)

# the non-zero rows in the data in column j
@inline _startcolrange((m,n), (l,u), j) = u-j+2:min(u+m+1-j,l+u+1)
@inline _bulkcolrange((m,n), (l,u), j) = 1:l+u+1
@inline _stopcolrange((m,n), (l,u), j) = 1:u+m+1-j

# how many rows we shift down
@inline _bulkshift((l,u), j) = j-u-1

@inline _startzcols(A) = _startzcols(size(A), bandwidths(A))
@inline _nzcols(A) = _nzcols(size(A), bandwidths(A))
@inline _stopzcols(A) = _stopzcols(size(A), bandwidths(A))
@inline _startcols(A) = _startcols(size(A), bandwidths(A))
@inline _bulkcols(A) = _bulkcols(size(A), bandwidths(A))
@inline _stopcols(A) = _stopcols(size(A), bandwidths(A))
@inline _startcolrange(A, j) = _startcolrange(size(A), bandwidths(A), j)
@inline _bulkcolrange(A, j) = _bulkcolrange(size(A), bandwidths(A), j)
@inline _stopcolrange(A, j) = _stopcolrange(size(A), bandwidths(A), j)
@inline _colrange(A, j) = _colrange(size(A), bandwidths(A), j)
@inline _colshift(A::AbstractMatrix, j) = _colshift(bandwidths(A), j)
@inline _bulkshift(A::AbstractMatrix, j) = _bulkshift(bandwidths(A), j)

@inline function _colrange((m,n), (l,u), j)
    j ≤ u && return _startcolrange((m,n), (l,u), j)
    j ≤ m-l && return _bulkcolrange((m,n), (l,u), j)
    return _stopcolrange((m,n), (l,u), j)
end
@inline function _colshift((l,u), j)
    j ≤ u && return 0
    _bulkshift((l,u),j)
end


function _intersectcolrange(kr_A,kr_d,sh)
    if sh ≤ 0
        kr_dA = first(kr_d):min(last(kr_d),first(kr_d)+length(kr_A)+sh-1)
        kr_Ad = range(first(kr_A)-sh; length=length(kr_dA))
    else
        kr_dA = first(kr_d)+sh:min(last(kr_d),first(kr_d)+length(kr_A)+sh-1)
        kr_Ad = range(first(kr_A); length=length(kr_dA))
    end
    kr_dA, kr_Ad
end

function _banded_broadcast!(dest::AbstractMatrix, f, A::AbstractMatrix{T}, ::BandedColumns, ::BandedColumns) where T
    z = f(zero(T))
    iszero(z) || checkbroadcastband(dest, size(A), bandwidths(broadcasted(f, A)))

    m,n = size(dest)
    (m == 0 || n == 0) && return dest
    data_d,data_A = bandeddata(dest), bandeddata(A)

    checkzerobands(dest, f, A)
    # copy top left entries
    jr = _bulkcols(A) ∩ _bulkcols(dest)
    for j = last(_startzcols(dest))+1:min(first(jr)-1,n)
        # don't bother optimising
        kr_A = _colrange(A,j)
        kr_d = _colrange(dest,j)
        sh = _colshift(A,j)-_colshift(dest,j)
        kr_dA,kr_Ad = _intersectcolrange(kr_A,kr_d,sh)
        if isempty(kr_Ad)
            data_d[kr_d,j] .= z
        else
            data_d[first(kr_d):first(kr_dA)-1,j] .= z
            data_d[kr_dA,j] .= f.(view(data_A,kr_Ad,j))
            data_d[last(kr_dA)+1:last(kr_d),j] .= z
        end
    end

    # the bulk
    kr_A = _bulkcolrange(A,first(jr))
    kr_d = _bulkcolrange(dest,first(jr))
    sh = _bulkshift(A,first(jr))-_bulkshift(dest,first(jr))
    kr_dA,kr_Ad = _intersectcolrange(kr_A,kr_d,sh)
    if isempty(kr_Ad)
        data_d[kr_d,jr] .= z
    else
        data_d[first(kr_d):first(kr_dA)-1,jr] .= z
        data_d[kr_dA,jr] .= f.(view(data_A,kr_Ad,jr))
        data_d[last(kr_dA)+1:last(kr_d),jr] .= z
    end

    # bottom right
    for j = last(jr)+1:min(last(_stopcols(dest)),n)
        kr_A = _colrange(A,j)
        kr_d = _colrange(dest,j)
        sh = _colshift(A,j)-_colshift(dest,j)
        kr_dA,kr_Ad = _intersectcolrange(kr_A,kr_d,sh)
        if isempty(kr_Ad)
            data_d[kr_d,j] .= z
        else
            data_d[first(kr_d):first(kr_dA)-1,j] .= z
            data_d[kr_dA,j] .= f.(view(data_A,kr_Ad,j))
            data_d[last(kr_dA)+1:last(kr_d),j] .= z
        end
    end

    dest
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractMatrix}})
    (A,) = bc.args
    _banded_broadcast!(dest, bc.f, A, MemoryLayout(typeof(dest)), MemoryLayout(typeof(A)))
end


###########
# matrix-number broadcast
###########

@inline function _banded_broadcast_loop_overlap!(dest, src, f::F, (min_sl_dl, min_su_du), m, j) where {F}
    for k = max(1,j-min_su_du):min(j+min_sl_dl,m)
        src_kj = inbands_getindex(src, k, j)
        v = f(src_kj)
        inbands_setindex!(dest, v, k, j)
    end
end

@inline function _banded_broadcast_loop!(dest, src, f, s_l_s_u, m)
    for j = rowsupport(dest)
        _banded_broadcast_loop_overlap!(dest, src, f, s_l_s_u, m, j)
    end
end

@inline function _banded_broadcast_loop!(dest, src, f, z, (s_l, s_u), (d_l, d_u), m)
    min_su_du = min(s_u, d_u)
    min_sl_dl = min(s_l, d_l)
    for j = rowsupport(dest)
        # if s_u < d_u, set extra bands in dest above the diagonal to zero
        for k = max(1,j-d_u):min(j+min(d_l, -(s_u+1)),m)
            inbands_setindex!(dest, z, k, j)
        end

        _banded_broadcast_loop_overlap!(dest, src, f, (min_sl_dl, min_su_du), m, j)

        # if s_l < d_l, set extra bands in dest below the diagonal to zero
        for k = max(1,j+max(s_l+1, -d_u)):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
end

@inline function _banded_broadcast_anylayout!(dest, src, f, z, (s_l, s_u), (d_l, d_u), m)
    if d_l == s_l && d_u == s_u
        _banded_broadcast_loop!(dest, src, f, (s_l, s_u), m)
    else
        _banded_broadcast_loop!(dest, src, f, z, (s_l, s_u), (d_l, d_u), m)
    end
end

function _banded_broadcast!(dest::AbstractMatrix, f, (src,x)::Tuple{AbstractMatrix{T},Number}, _1, _2) where {T}
    z = f(zero(T), x)
    iszero(z) || checkbroadcastband(dest, size(src), bandwidths(broadcasted(f, src,x)))
    m,n = size(dest)
    m == n == 0 && return dest

    f_x = Base.Fix2(f, x)

    # if dest has fewer bands (either above or below the diagonal) than dest,
    # then f.(x, dest) must be zero in these bands
    checkzerobands(dest, f_x, src)

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)

    _banded_broadcast_anylayout!(dest, src, f_x, z, (s_l, s_u), (d_l, d_u), m)

    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (x,src)::Tuple{Number,AbstractMatrix{T}}, _1, _2) where T
    z = f(x, zero(T))
    iszero(z) || checkbroadcastband(dest, size(src), bandwidths(broadcasted(f, x,src)))
    m,n = size(dest)
    m == n == 0 && return dest

    f_x = Base.Fix1(f, x)

    # if dest has fewer bands (either above or below the diagonal) than dest,
    # then f.(x, src) must be zero in these bands
    checkzerobands(dest, f_x, src)

    d_l, d_u = bandwidths(dest)
    s_l, s_u = bandwidths(src)

    _banded_broadcast_anylayout!(dest, src, f_x, z, (s_l, s_u), (d_l, d_u), m)

    dest
end

function _banded_broadcast_BandedColumn!(dest, f, src, z)
    l,u = bandwidths(src)
    λ,μ = bandwidths(dest)
    m,n = size(src)
    data_d,data_s = bandeddata(dest), bandeddata(src)
    d_m, d_n = size(data_d)

    if (l,u) == (λ,μ)
        data_d .= f.(data_s)
    elseif μ > u && λ > l
        fill!(view(data_d,1:min(μ-u,d_m),:), z)
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,μ-u+1:μ+l+1,:) .= f.(data_s)
    elseif μ > u
        fill!(view(data_d,1:min(μ-u,d_m),:), z)
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        view(data_d,μ-u+1:μ+λ+1,:) .= f.(view(data_s,1:u+λ+1,:))
    elseif λ > l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        fill!(view(data_d,μ+l+2:μ+λ+1,:), z)
        view(data_d,1:μ+l+1,:) .= f.(view(data_s,u-μ+1:u+l+1,:))
    else # μ < u && λ < l
        for b = μ+1:u
            any(!iszero, view(data_s,u-b+1,b+1:n)) && throw(BandError(dest,b))
        end
        for b = λ+1:l
            any(!iszero, view(data_s,u+b+1,1:min(m-b,n))) && throw(BandError(dest,b))
        end
        data_d .= f.(view(data_s,u-μ+1:u+λ+1,:))
    end
    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (src,x)::Tuple{AbstractMatrix{T},Number}, ::BandedColumns, ::BandedColumns) where T
    z = f(zero(T),x)
    iszero(z) || checkbroadcastband(dest, size(src), bandwidths(broadcasted(f, src,x)))

    f_x = Base.Fix2(f, x)
    _banded_broadcast_BandedColumn!(dest, f_x, src, z)

    dest
end

function _banded_broadcast!(dest::AbstractMatrix, f, (x, src)::Tuple{Number,AbstractMatrix{T}}, ::BandedColumns, ::BandedColumns) where T
    z = f(x, zero(T))
    iszero(z) || checkbroadcastband(dest, size(src), bandwidths(broadcasted(f, x,src)))

    f_x = Base.Fix1(f, x)
    _banded_broadcast_BandedColumn!(dest, f_x, src, z)

    dest
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractMatrix,Number}})
    (A,x) = bc.args
    _banded_broadcast!(dest, bc.f, (A, x), MemoryLayout(typeof(dest)), MemoryLayout(typeof(A)))
end

function copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{Number,AbstractMatrix}})
    (x,A) = bc.args
    _banded_broadcast!(dest, bc.f, (x,A), MemoryLayout(typeof(dest)), MemoryLayout(typeof(A)))
end

###############
# matrix-vector broadcast
###############

_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractVector{T},AbstractMatrix{V}}, _1, _2) where {T,V} =
    _left_colvec_banded_broadcast!(dest, f, (A,B), _1, _2)

function _left_colvec_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractVecOrMat{T},AbstractMatrix{V}}, _1, _2) where {T,V}
    z = f(zero(T), zero(V))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    A_l, A_u = _broadcast_bandwidths((m-1,n-1),A)
    @assert A_u == n-1
    B_l, B_u = bandwidths(B)
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    if d_l == B_l == l && d_u == B_u == u
        for j=rowsupport(dest)
            for k = max(1,j-u):min(j+min(A_l,l),m)
                inbands_setindex!(dest, f(A[k], inbands_getindex(B, k, j)), k, j)
            end
            for k = max(1,j-u,j+A_l+1):min(j+l,m)
                inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
            end
        end
    else
        for j=rowsupport(dest)
            for k = max(1,j-d_u):min(j-u-1,m)
                inbands_setindex!(dest, z, k, j)
            end
            for k = max(1,j-d_u):min(j-B_u-1,j+d_l,m)
                inbands_setindex!(dest, f(A[k], zero(V)), k, j)
            end
            for k = max(1,j-B_u):min(j+min(A_l,B_l),m)
                inbands_setindex!(dest, f(A[k], inbands_getindex(B, k, j)), k, j)
            end
            for k = max(1,j-d_u,j+B_l+1):min(j+A_l,j+d_l,m)
                inbands_setindex!(dest, f(A[k], zero(V)), k, j)
            end
            for k = max(1,j-d_u,j+A_l+1):min(j+B_l,j+d_l,m)
                inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
            end
            for k = max(1,j+l+1):min(j+d_l,m)
                inbands_setindex!(dest, z, k, j)
            end
        end
    end
    dest
end

_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractVector{V}}, _1, _2) where {T,V} =
    _right_colvec_banded_broadcast!(dest, f, (A,B), _1, _2)

function _right_colvec_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractVecOrMat{V}}, _1, _2) where {T,V}
    z = f(zero(T), zero(V))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    A_l, A_u = bandwidths(A)
    B_l, B_u = _broadcast_bandwidths((m-1,n-1),B)
    @assert B_u == n-1
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    if d_l == A_l == l && d_u == A_u == u
        for j=rowsupport(dest)
            for k = max(1,j-u):min(j+min(l,B_l),m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), B[k]), k, j)
            end
            for k = max(1,j-u,j+B_l+1):min(j+l,m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
            end
        end
    else
        for j=rowsupport(dest)
            for k = max(1,j-d_u):min(j-u-1,m)
                inbands_setindex!(dest, z, k, j)
            end
            for k = max(1,j-d_u):min(j-A_u-1,j+d_l,m)
                inbands_setindex!(dest, f(zero(T), B[k]), k, j)
            end
            for k = max(1,j-A_u):min(j+min(A_l,B_l),m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), B[k]), k, j)
            end
            for k = max(1,j-d_u,j+B_l+1):min(j+A_l,j+d_l,m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
            end
            for k = max(1,j-d_u,j+A_l+1):min(j+B_l,j+d_l,m)
                inbands_setindex!(dest, f(zero(T), B[k]), k, j)
            end
            for k = max(1,j+l+1):min(j+d_l,m)
                inbands_setindex!(dest, z, k, j)
            end
        end
    end
    dest
end

function __left_rowvec_banded_broadcast!(dest, f, (A,B),
        ::BandedColumns,
        ::Tuple{Union{DenseColumnMajor, DualLayout{ArrayLayouts.DenseRowMajor}}, BandedColumns},
        (l, u), (A_l,A_u), (m,n))

    D = bandeddata(dest)
    Bd = bandeddata(B)
    D .= f.(A, Bd)
    return nothing
end

function __left_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2, (l, u), (A_l,A_u), (m,n))
    for j=rowsupport(dest)
        for k = max(1,j-u):min(j-A_u-1,j+l,m)
            inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j-min(A_u,u)):min(j+l,m)
            inbands_setindex!(dest, f(A[j], inbands_getindex(B, k, j)), k, j)
        end
    end
end

function _left_rowvec_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _1, _2) where {T,V}
    @assert size(A,1) == 1
    z = f(zero(T), zero(V))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    A_l, A_u = _broadcast_bandwidths((m-1,n-1),A)
    @assert A_l == m-1
    B_l, B_u = bandwidths(B)
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    if d_l == B_l == l && d_u == B_u == u
        __left_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2,
                (l, u), (A_l,A_u), (m,n))
    else
        for j=rowsupport(dest)
            for k = max(1,j-d_u):min(j-u-1,m)
                inbands_setindex!(dest, z, k, j)
            end
            for k = max(1,j-d_u,j-A_u):min(j-B_u-1,j+d_l,m)
                inbands_setindex!(dest, f(A[j], zero(V)), k, j)
            end
            for k = max(1,j-d_u,j-B_u):min(j-A_u-1,j+d_l,m)
                inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
            end
            for k = max(1,j-min(A_u,B_u)):min(j+min(A_l,B_l),m)
                inbands_setindex!(dest, f(A[j], inbands_getindex(B, k, j)), k, j)
            end
            for k = max(1,j-d_u,j+B_l+1):min(j+d_l,m)
                inbands_setindex!(dest, f(A[j], zero(V)), k, j)
            end
            for k = max(1,j+l+1):min(j+d_l,m)
                inbands_setindex!(dest, z, k, j)
            end
        end
    end
    dest
end

function __right_rowvec_banded_broadcast!(dest, f, (A,B),
        ::BandedColumns,
        ::Tuple{BandedColumns, Union{DualLayout{ArrayLayouts.DenseRowMajor}, DenseColumnMajor}},
        (l, u), (B_l,B_u), (m,n))

    D = bandeddata(dest)
    Ad = bandeddata(A)
    D .= f.(Ad, B)
    return nothing
end

function __right_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2, (l, u), (B_l,B_u), (m,n))
    for j=rowsupport(dest)
        for k = max(1,j-u):min(j-B_u-1,j+l,m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
        end
        for k = max(1,j-min(u,B_u)):min(j+l,m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), B[j]), k, j)
        end
    end
end

function _right_rowvec_banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _1, _2) where {T,V}
    @assert size(B,1) == 1
    z = f(zero(T), zero(V))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))
    m,n = size(dest)

    d_l, d_u = bandwidths(dest)
    A_l, A_u = bandwidths(A)
    B_l, B_u = _broadcast_bandwidths((m-1,n-1),B)
    @assert B_l ==  m-1
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    if d_l == A_l == l && d_u == A_u == u
        __right_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2,
                (l, u), (B_l,B_u), (m,n))
    else
        for j=rowsupport(dest)
            for k = max(1,j-d_u):min(j-u-1,m)
                inbands_setindex!(dest, z, k, j)
            end
            for k = max(1,j-d_u,j-A_u):min(j-B_u-1,j+d_l,m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
            end
            for k = max(1,j-d_u,j-B_u):min(j-A_u-1,j+d_l,m)
                inbands_setindex!(dest, f(zero(T), B[j]), k, j)
            end
            for k = max(1,j-min(A_u,B_u)):min(j+A_l,m)
                inbands_setindex!(dest, f(inbands_getindex(A, k, j), B[j]), k, j)
            end
            for k = max(1,j-d_u,j+A_l+1):min(j+d_l,m)
                inbands_setindex!(dest, f(zero(T), B[j]), k, j)
            end
            for k = max(1,j+l+1):min(j+d_l,m)
                inbands_setindex!(dest, z, k, j)
            end
        end
    end
    dest
end


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractVector,AbstractMatrix}}) =
    _banded_broadcast!(dest, bc.f, bc.args, MemoryLayout(dest), MemoryLayout.(bc.args))


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractMatrix,AbstractVector}}) =
    _banded_broadcast!(dest, bc.f, bc.args, MemoryLayout(dest), MemoryLayout.(bc.args))


################
# matrix-matrix broadcast
################

function _banded_broadcast!(dest::AbstractMatrix, f, (A,B)::NTuple{2,AbstractMatrix}, _1, _2)
    __banded_broadcast!(dest, f, (A,B), _1, _2)
end
function __banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _1, _2) where {T,V}
    z = f(zero(T), zero(V))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))
    m,n = size(dest)
    if size(A) ≠ (m,n)
        size(A,2) == 1 && return _left_colvec_banded_broadcast!(dest, f, (A,B), _1, _2)
        return _left_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2)
    end
    if size(B) ≠ (m,n)
        size(B,2) == 1 && return _right_colvec_banded_broadcast!(dest, f, (A,B), _1, _2)
        return _right_rowvec_banded_broadcast!(dest, f, (A,B), _1, _2)
    end

    d_l, d_u = bandwidths(dest)
    A_l, A_u = bandwidths(A)
    B_l, B_u = bandwidths(B)
    (d_l ≥ min(l,m-1) && d_u ≥ min(u,n-1)) || throw(BandError(dest))

    for j=rowsupport(dest)
        for k = max(1,j-d_u):min(j-u-1,j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-min(A_u,d_u)):min(j-B_u-1,j+min(A_l,d_l),m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
        end
        for k = max(1, j+A_l+1, j-d_u):min(j-B_u-1,j+d_l,m)
            # This is hit in A+B with bandwidth(A) == (-2,2) && bandwidth(B) == (0,0)
            # The result has a bandwidth (2,0). This sets dest[band(1)] to zero
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j-min(B_u,d_u)):min(j-A_u-1,j+min(B_l,d_l),m)
            inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j-min(A_u,B_u,d_u)):min(j+min(A_l,B_l,d_l),m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j+B_l+1,j-min(A_u,d_u)):min(j+min(A_l,d_l),m)
            inbands_setindex!(dest, f(inbands_getindex(A, k, j), zero(V)), k, j)
        end
        for k = max(1, j+B_l+1, j-d_u):min(j-A_u-1,j+d_l,m)
            # This is hit in A + B with bandwidth(A) == (2,-2) && bandwidth(B) == (0,0)
            # The result has a bandwidth (2,0). This sets dest[band(-1)] to zero
            inbands_setindex!(dest, z, k, j)
        end
        for k = max(1,j+A_l+1,j-min(d_u, B_u)):min(j+min(B_l,d_l),m)
            inbands_setindex!(dest, f(zero(T), inbands_getindex(B, k, j)), k, j)
        end
        for k = max(1,j-d_u,j+l+1):min(j+d_l,m)
            inbands_setindex!(dest, z, k, j)
        end
    end
    dest
end

function checkzerobands(dest, f, (A,B)::NTuple{2,AbstractMatrix})
    m,n = size(A)
    d_l, d_u = bandwidths(dest)
    A_l, A_u = bandwidths(A)
    B_l, B_u = bandwidths(B)
    l, u = max(A_l,B_l), max(A_u,B_u)

    rspA = rowsupport(A)
    rspB = rowsupport(B)
    for j = min(minimum(rspA), minimum(rspB)):max(maximum(rspA), maximum(rspB))
        for k = max(1,j-u) : min(j-d_u-1,m)
            iszero(f(A[k,j], B[k,j])) || throw(BandError(dest,b))
        end
        for k = max(1,j+d_l+1) : min(j+l,m)
            iszero(f(A[k,j], B[k,j])) || throw(BandError(dest,b))
        end
    end
end


function _banded_broadcast!(dest::AbstractMatrix, f, (A,B)::NTuple{2,AbstractMatrix}, Mdest::BandedColumns, MAB::NTuple{2,BandedColumns})
    z = f(zero(eltype(A)), zero(eltype(B)))
    bc = broadcasted(f, A, B)
    l, u = bandwidths(bc)
    (-u ≥ size(dest,1) || -l ≥ size(dest,2)) && return fill!(dest, z)
    iszero(z) || checkbroadcastband(dest, size(bc), (l,u))

    if size(A) ≠ size(dest) || size(B) ≠ size(dest)
        # special broadcast
        return __banded_broadcast!(dest, f, (A,B), Mdest, MAB)
    end

    A_l,A_u = bandwidths(A)
    B_l,B_u = bandwidths(B)
    -A_l > A_u && return broadcast!(f, dest, zero(eltype(A)), B)
    -B_l > B_u && return broadcast!(f, dest, A, zero(eltype(B)))
    d_l,d_u = bandwidths(dest)
    m,n = size(A)
    data_d,data_A, data_B = bandeddata(dest), bandeddata(A), bandeddata(B)

    if (d_l,d_u) == (A_l,A_u) == (B_l,B_u)
        data_d .= f.(data_A,data_B)
    else
        max_l,max_u = max(A_l,B_l,d_l),max(A_u,B_u,d_u)
        min_l,min_u = min(A_l,B_l,d_l),min(A_u,B_u,d_u)
        checkzerobands(dest, f, (A,B))

        # fill extra bands in dest
        fill!(view(data_d,1:d_u-max(A_u,B_u),:), z)
        fill!(view(data_d,d_u+max(A_l,B_l)+2:size(data_d,1),:), z)
        # between A and B if non-overlapping
        bs = max(-d_u,B_l+1):min(-A_u-1,d_l)
        fill!(view(data_d, bs .+ d_u .+ 1, :), z)
        bs = max(-d_u,A_l+1):min(-B_u-1,d_l)
        fill!(view(data_d, bs .+ d_u .+ 1, :), z)

        # construct where B upper is zero
        # this is from band A_u:B_u+1
        bs = max(-d_u,-A_u):min(-1-B_u,A_l,d_l)
        data_d_u_A = view(data_d, bs .+ d_u .+ 1, :)
        data_A_u_A = view(data_A, bs .+ A_u .+ 1, :)
        data_d_u_A .= f.(data_A_u_A, zero(eltype(B)))

        # construct where A upper is zero
        # this is from band B_u:min(A_u+1,-B_l)
        bs = max(-d_u,-B_u):min(-1-A_u,B_l,d_l)
        data_d_u_B = view(data_d, bs .+ d_u .+ 1, :)
        data_B_u_B = view(data_B, bs .+ B_u .+ 1, :)
        data_d_u_B .= f.(zero(eltype(A)), data_B_u_B)

        # construct where A and B are non-zero
        data_d̃ = view(data_d, d_u-min_u+1 : d_u+min_l+1, :)
        data_Ã = view(data_A, A_u-min_u+1 : A_u+min_l+1, :)
        data_B̃ = view(data_B, B_u-min_u+1 : B_u+min_l+1, :)
        data_d̃ .= f.(data_Ã,data_B̃)

        # construct where A lower is zero
        bs = max(A_l+1,-B_u,-d_u):min(B_l,d_l)
        data_d_l_B = view(data_d, bs .+ d_u .+ 1, :)
        data_B_l_B = view(data_B, bs .+ B_u .+ 1, :)
        data_d_l_B .= f.(zero(eltype(A)), data_B_l_B)

        # construct where B lower is zero
        bs = max(B_l+1,-A_u,-d_u):min(A_l,d_l)
        data_d_l_A = view(data_d, bs .+ d_u .+ 1, :)
        data_A_l_A = view(data_A, bs .+ A_u .+ 1, :)
        data_d_l_A .= f.(data_A_l_A, zero(eltype(B)))
    end

    dest
end


copyto!(dest::AbstractArray, bc::Broadcasted{BandedStyle, <:Any, <:Any, <:NTuple{2,AbstractMatrix}}) =
    _banded_broadcast!(dest, bc.f, bc.args, MemoryLayout(typeof(dest)), map(MemoryLayout,bc.args))

# override copy in case data has special broadcast
_default_banded_broadcast(bc::Broadcasted{Style}, _) where Style = Base.invoke(copy, Tuple{Broadcasted{Style}}, bc)
_default_banded_broadcast(bc::Broadcasted) = _default_banded_broadcast(bc, axes(bc))

_banded_broadcast(f, args::Tuple, _...) = _default_banded_broadcast(broadcasted(f, args...))
_banded_broadcast(f, arg, _...) = _default_banded_broadcast(broadcasted(f, arg))

function _banded_broadcast(f, A::AbstractMatrix{T}, ::BandedColumns) where T
    iszero(f(zero(T))) || return _default_banded_broadcast(broadcasted(f, A))
    Bdata = bandeddata(A)
    Bdata_new = reshape(f.(vec(Bdata)), size(Bdata))
    _BandedMatrix(Bdata_new, axes(A,1), bandwidths(A)...)
end
function _banded_broadcast(f, (src,x)::Tuple{AbstractMatrix{T},Number}, ::BandedColumns) where T
    iszero(f(zero(T),x)) || return _default_banded_broadcast(broadcasted(f, src,x))
    Bdata = bandeddata(src)
    Bdata_new = reshape(f.(vec(Bdata), x), size(Bdata))
    _BandedMatrix(Bdata_new, axes(src,1), bandwidths(src)...)
end
function _banded_broadcast(f, (x, src)::Tuple{Number,AbstractMatrix{T}}, ::BandedColumns) where T
    iszero(f(x, zero(T))) || return _default_banded_broadcast(broadcasted(f, x,src))
    Bdata = bandeddata(src)
    Bdata_new = reshape(f.(x, vec(Bdata)), size(Bdata))
    _BandedMatrix(Bdata_new, axes(src,1), bandwidths(src)...)
end

function copy(bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractMatrix}})
    (A,) = bc.args
    _banded_broadcast(bc.f, A, MemoryLayout(typeof(A)))
end

function copy(bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{AbstractMatrix,Number}})
    (A,x) = bc.args
    _banded_broadcast(bc.f, (A, x), MemoryLayout(typeof(A)))
end


function copy(bc::Broadcasted{BandedStyle, <:Any, <:Any, <:Tuple{Number,AbstractMatrix}})
    (x,A) = bc.args
    _banded_broadcast(bc.f, (x,A), MemoryLayout(typeof(A)))
end


function copy(bc::Broadcasted{BandedStyle, <:Any, <:Any, <:NTuple{2,AbstractMatrix}})
    _banded_broadcast(bc.f, bc.args, MemoryLayout.(typeof.(bc.args)))
end

###
# Special cases
###

#=
    B::BandedMatrix * D::Diagonal for DenseColumnMajor storage may be evaluated as
    B.data .* D.diag', instead of B .* D.diag'
=#
function copy(M::Rmul{BandedColumns{DenseColumnMajor}, DiagonalLayout{DenseColumnMajor}})
    B, D = M.A, M.B
    d = diagonaldata(D)
    Bd = bandeddata(B)
    Bdata = if size(Bd, 1) == 1
                # performance optimization
                # may not be necessary in the future if the other branch
                # is equally fast
                reshape(vec(Bd) .* d, 1, :)
            else
                Bd .* d'
            end
    _BandedMatrix(Bdata, size(B,1), bandwidths(B)...)
end

###
# broadcast bandwidths
###


_bcsize(A) = size(A)


_broadcast_bandwidths(bnds) = bnds
_broadcast_bandwidths(bnds, _) = bnds
_broadcast_bandwidths((l,u), a::AbstractVector) = (bandwidth(a,1),u)
function __broadcast_bandwidths((l,u), A)
    sz = _bcsize(A)
    (length(sz) == 1 || sz[2] == 1) && return (bandwidth(A,1),u)
    sz[1] == 1 && return (l, bandwidth(A,2))
    bandwidths(A) # need to special case vector broadcasting
end
_broadcast_bandwidths((l,u), A::AbstractArray) = __broadcast_bandwidths((l,u), A)
_broadcast_bandwidths((l,u), A::Broadcasted) = __broadcast_bandwidths((l,u), A)


_band_eval_args() = ()
_band_eval_args(a, b...) = (a, _band_eval_args(b...)...)
_band_eval_args(::Base.RefValue{Type{T}}, b...) where T = (T, _band_eval_args(b...)...)
_band_eval_args(a::AbstractMatrix{T}, b...) where T = (zero(T), _band_eval_args(b...)...)
_band_eval_args(a::AbstractVector{T}, b...) where T = (one(T), _band_eval_args(b...)...)
_band_eval_args(a::Broadcasted, b...) = (zero(Base.Broadcast.combine_eltypes(a.f, a.args)), _band_eval_args(b...)...)


# zero dominates. Take the minimum bandwidth. Always treat like matrix.
__bnds(m, n) = (m-1, n-1)
__bnds(m) = (m-1, 0)
_bnds(bc) = __bnds(size(bc)...)

bandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(*)}) =
    min.(_broadcast_bandwidths.(Ref(_bnds(bc)), bc.args)...)

bandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(/)}) = _broadcast_bandwidths(_bnds(bc), first(bc.args))
bandwidths(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Any,typeof(\)}) = _broadcast_bandwidths(_bnds(bc), last(bc.args))


# zero is preserved. Take the maximum bandwidth
_isweakzero(f, args...) =  iszero(f(_band_eval_args(args...)...))


function bandwidths(bc::Broadcasted)
    sz = _bcsize(bc)
    a,b = length(sz) == 1 ? (sz[1],1) : sz
    bnds = (a-1,b-1)
    _isweakzero(bc.f, bc.args...) && return min.(bnds, max.(_broadcast_bandwidths.(Ref(bnds), bc.args)...))
    bnds
end

banded_similar(T, sz, bnds) = BandedMatrix{T}(undef, sz, bnds)
similar(bc::Broadcasted{BandedStyle}, ::Type{T}) where T = banded_similar(T, size(bc), bandwidths(bc))



##
# lmul!/rmul!
##

function _banded_lmul!(α, A::AbstractMatrix, _)
    for j = rowsupport(A), k = colsupport(A,j)
        inbands_setindex!(A, α*inbands_getindex(A,k,j), k,j)
    end
    A
end

function _banded_rmul!(A::AbstractMatrix, α, _)
    for j = rowsupport(A), k = colsupport(A,j)
        inbands_setindex!(A, inbands_getindex(A,k,j)*α, k,j)
    end
    A
end


function _banded_lmul!(α::Number, A::AbstractMatrix, ::BandedColumns)
    lmul!(α, bandeddata(A))
    A
end

function _banded_rmul!(A::AbstractMatrix, α::Number, ::BandedColumns)
    rmul!(bandeddata(A), α)
    A
end

banded_lmul!(α::Number, A::AbstractMatrix) = _banded_lmul!(α, A, MemoryLayout(typeof(A)))
banded_rmul!(A::AbstractMatrix, α::Number) = _banded_rmul!(A, α, MemoryLayout(typeof(A)))

materialize!(L::Lmul{ScalarLayout,<:AbstractBandedLayout}) = banded_lmul!(L.A, L.B)
materialize!(L::Rmul{<:AbstractBandedLayout,ScalarLayout}) = banded_rmul!(L.A, L.B)

##
# axpy!
##

# these are the routines of the banded interface of other AbstractMatrices
banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix) = _banded_axpy!(a, X, Y, MemoryLayout(typeof(X)), MemoryLayout(typeof(Y)))
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, ::BandedColumns, ::BandedColumns) =
    banded_generic_axpy!(a, X, Y)
_banded_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix, notbandedX, notbandedY) =
    banded_dense_axpy!(a, X, Y)

# additions and subtractions
@propagate_inbounds function banded_generic_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    n,m = size(X)
    if (n,m) ≠ size(Y)
        throw(BoundsError())
    end
    Xl, Xu = bandwidths(X)
    Yl, Yu = bandwidths(Y)

    if -Xl > Xu #= no bands in X =#
        return Y
    end

    @boundscheck if Xl > Yl
        # test that all entries are zero in extra bands below the diagonal
        for j=rowsupport(X),k=max(1,j+Yl+1):min(j+Xl,n)
            if inbands_getindex(X, k, j) ≠ 0
                throw(BandError(Y, (k,j)))
            end
        end
    end
    @boundscheck if Xu > Yu
        # test that all entries are zero in extra bands above the diagonal
        for j=rowsupport(X),k=max(1,j-Xu):min(j-Yu-1,n)
            if inbands_getindex(X, k, j) ≠ 0
                throw(BandError(Y, (k,j)))
            end
        end
    end

    if -Yl > Yu #= no bands in Y =#
        return Y
    end

    # only fill overlapping bands
    l = min(Xl,Yl)
    u = min(Xu,Yu)

    @inbounds for j=rowsupport(X), k=max(1,j-u):min(n,j+l)
        inbands_setindex!(Y, a*inbands_getindex(X,k,j) + inbands_getindex(Y,k,j) ,k, j)
    end
    Y
end

function banded_dense_axpy!(a::Number, X::AbstractMatrix, Y::AbstractMatrix)
    if size(X) != size(Y)
        throw(DimensionMismatch("+"))
    end
    @inbounds for j=rowsupport(X), k=colrange(X,j)
        Y[k,j] += a*inbands_getindex(X,k,j)
    end
    Y
end

##
# special cases to hack tests where 2.0 .* A .+ B can be constructed in different ways depending on REPL or in a function.
# Should be made more general.
##
function copyto!(dest::AbstractArray{T}, bc::Broadcasted{BandedStyle, <:Any, typeof(+),
            <:Tuple{
                Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{Number,AbstractMatrix}},
                AbstractMatrix}}) where T
    αA,B = bc.args
    α,A = αA.args
    dest ≡ B || (dest .= B)
    banded_axpy!(α, A, dest)
end

function similar(bc::Broadcasted{BandedStyle, <:Any, typeof(+),
            <:Tuple{
                Broadcasted{BandedStyle,<:Any,typeof(*),<:Tuple{Number,AbstractMatrix}},
                AbstractMatrix}}, ::Type{T}) where T
    αA,B = bc.args
    α,A = αA.args
    n,m = size(A)
    (n,m) == size(B) || throw(DimensionMismatch())
    Al,Au = bandwidths(A)
    Bl,Bu = bandwidths(B)
    similar(A, T, n, m, max(Al,Bl), max(Au,Bu))
end

axpy!(α, A::AbstractBandedMatrix, dest::AbstractMatrix) = banded_axpy!(α, A, dest)


###
# Fill
###

for op in (:*, :/)
    @eval begin
        broadcasted(::BandedStyle, ::typeof($op), a::Zeros, b::AbstractArray) = _broadcasted_zeros(a, b)
        function broadcasted(::BandedStyle, ::typeof($op), a::AbstractArray{T}, b::Ones{V}) where {T,V}
            Base.Broadcast.combine_axes(a, b) # dimension check
            _copy_oftype(a, promote_op(*, T, V))
        end
    end
end

for op in (:*, :\)
    @eval begin
        broadcasted(::BandedStyle, ::typeof($op), a::AbstractArray, b::Zeros) = _broadcasted_zeros(a, b)
        function broadcasted(::BandedStyle, ::typeof($op), a::Ones{T}, b::AbstractArray{V}) where {T,V}
            Base.Broadcast.combine_axes(a, b) # dimension check
            _copy_oftype(b, promote_op(*, T, V))
        end
    end
end
