"""
    tridiagonalize(A, d::Integer=0)

The symmetric real or Hermitian input matrix `A` is transformed to a real
`SymTridiagonal` matrix.
For general matrices only `d` superdiagonals are processed.
Works efficiently for `BandedMatrices`.
"""
function tridiagonalize(A::Union{Symmetric{<:Real},Hermitian}, d::Integer=0)
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $((m, n))"))
    dmax = bandwidth(A) + 1
    if d == 0
        d = min(n, dmax)
    end

    0 < d <= dmax || throw(ArgumentError("number of diagonals $d not in 1:$dmax"))

    B = copybands(A, d)
    td = _tridiag_algorithm!(B)
    dv = [real(B[1,i]) for i in 1:n]
    ev = d > 1 ? [-abs(B[2,i]) for i = 2:n] : zeros(real(eltype(A)), n-1)
    SymTridiagonal(dv, ev)
end

# B[1:q,1:n] has the bands of a (2q-1)-banded hermitian nxn-matrix.
# superdiagonal i valid in B[i,i+1:n]. B is completely overwritten.
# The result is in B[1:2,1:nvens2]
function _tridiag_algorithm!(B)
    d, n = size(B)
    0 < d <= n || throw(ArgumentError("number of diagonals $d not in 1:$n"))

    @inbounds for bm = d-1:-1:2
        for k = 1:n-bm
            kp = k
            apiv = B[bm+1,bm+kp]
            iszero(apiv) && continue
            for i = bm+k-1:bm:n-1
                b = B[ i-kp+1,i]
                c, s, r = LinearAlgebra.givensAlgorithm(b, apiv)
                u, v = B[1,i], B[1,i+1]
                upx = (u + v) / 2
                B[1,i] = (u - v) / 2
                B[i-kp+1,i] = r
                for j = kp+1:i
                    u = B[i-j+1,i]
                    v = B[i-j+2,i+1]
                    B[i-j+1,i], B[i-j+2,i+1] = u * c + v * s, -u * s' + v * c
                end
                B[1,i+1] = -(B[1,i])'
                ip = i + bm
                for j = i+1:min(ip, n)
                    u = B[j-i+1,j]
                    v = B[j-i,j]
                    B[j-i+1,j], B[j-i,j] = u * c + v * s', -u * s + v * c
                end
                w = real(B[1,i+1])
                B[1,i] = upx - w
                B[1,i+1] = upx + w
                if ip < n
                    v = B[ip-i+1,ip+1]
                    apiv, B[ip-i+1,ip+1] = v * s', v * c
                end
                kp = i
            end
        end
    end
    B
end

# generalization of method for symmetric BandedMatrices
bandwidth(A::AbstractMatrix) = min(size(A)...) - 1

function copybands(A::AbstractMatrix{T}, d::Integer) where T
    n = min(size(A)...)
    d = min(d, n)
    B = Matrix{T}(undef, d, n)
    for i = 1:d
        B[i,1:i-1] .= zero(T)
        for j = i:n
            B[i,j] = A[j-i+1,j]
        end
    end
    B
end

function _tridiagonalize!(A::AbstractMatrix{T}, ::SymmetricLayout{<:BandedColumnMajor}) where T<:BlasReal
    n=size(A, 1)
    d = Vector{T}(undef,n)
    e = Vector{T}(undef,n-1)
    Q = Matrix{T}(undef,0,0)
    work = Vector{T}(undef,n)
    sbtrd!('N', symmetricuplo(A), size(A,1), bandwidth(A), symbandeddata(A), d, e, Q, work)
    SymTridiagonal(d,e)
end

function _tridiagonalize!(A::AbstractMatrix{T}, ::HermitianLayout{<:BandedColumnMajor}) where T<:BlasComplex
    n=size(A, 1)
    d = Vector{real(T)}(undef,n)
    e = Vector{real(T)}(undef,n-1)
    Q = Matrix{T}(undef,0,0)
    work = Vector{T}(undef,n)
    hbtrd!('N', symmetricuplo(A), size(A,1), bandwidth(A), hermbandeddata(A), d, e, Q, work)
    SymTridiagonal(d,e)
end

tridiagonalize!(A::AbstractMatrix{<:BlasFloat}) = _tridiagonalize!(A, MemoryLayout(typeof(A)))
