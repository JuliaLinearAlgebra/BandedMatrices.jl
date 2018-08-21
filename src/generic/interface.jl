



# matrix * vector


# matrix * matrix
banded_matmatmul!(C::AbstractMatrix, tA::Val, tB::Val, A::AbstractMatrix, B::AbstractMatrix) =
    banded_matmatmul!(C, tA, tB, A, B, MemoryLayout(A), MemoryLayout(B))
banded_matmatmul!(C::AbstractMatrix, tA::Val, tB::Val, A::AbstractMatrix, B::AbstractMatrix,
                  notblasA, notblasB) =
    banded_generic_matmatmul!(C, tA, tB, A, B)
banded_matmatmul!(C::AbstractMatrix, tA::Val, tB::Val, A::AbstractMatrix, B::AbstractMatrix,
                  ::BandedColumnMajor, ::BandedColumnMajor) =
    generally_banded_matmatmul!(C, tA, tB, A, B)



# some default functions




# matrix * matrix
function _banded_generic_matmatmul!(C::AbstractMatrix{T}, tA::Val{'N'}, tB::Val{'N'}, A::AbstractMatrix{U}, B::AbstractMatrix{V}) where {T, U, V}
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
            tmp = tmp + inbands_getindex(A,k,ν) * inbands_getindex(B,ν,j)
        end
        inbands_setindex!(C,tmp,k,j)
    end
    C
end
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



# add banded linear algebra routines between Typ1 and Typ2 both implementing
# the BandedMatrix interface
macro _banded_banded_linalg(Typ1, Typ2)
    ret = quote
        Base.copyto!(dest::$Typ1, src::$Typ2) = BandedMatrices.banded_copyto!(dest,src)
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


        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ1, B::$Typ2) = banded_matmatmul!(C, Val{'N'}(), Val{'N'}(), A, B)

        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ1{T}}, B::$Typ2) where T = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ1, Bc::Adjoint{U,<:$Typ2{U}}) where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'C'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ1{T}}, Bc::Adjoint{U,<:$Typ2{U}}) where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'C'}(), parent(Ac), parent(Bc))

        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:$Typ1{T}}, B::$Typ2) where T = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(At), B)
        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ1, Bt::Transpose{U,<:$Typ2{U}}) where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bt))
        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:$Typ1{T}}, Bt::Transpose{U,<:$Typ2{U}}) where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(At), parent(Bt))

        # override mul! for real values
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{U,<:$Typ1{U}}, B::$Typ2{V}) where {U<:Real, V} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ1{U}, Bc::Adjoint{V,<:$Typ2{V}}) where {U, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ1{T}}, Bc::Adjoint{U,<:$Typ2{U}}) where {T<:Real,U} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'C'}(), parent(Ac), parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ1{T}}, Bc::Adjoint{U,<:$Typ2{U}}) where {T,U<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'T'}(), parent(Ac), parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{U,<:$Typ1{U}}, Bc::$Adjoint{V,<:$Typ2{V}}) where {U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(Ac), parent(Bc))

        function Base.:*(A::$Typ1{T}, B::$Typ2{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            Y = BandedMatrix{promote_type(T,V)}(undef, n, m, prodbandwidths(A, B)...)
            mul!(Y, A, B)
        end

        Base.:*(Ac::Adjoint{T,<:$Typ1{T}}, B::$Typ2{V}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'C'}(), Val{'N'}(), parent(Ac), B, promote_type(T, V)), Ac, B)
        Base.:*(A::$Typ1{T}, Bc::Adjoint{V,<:$Typ2{V}}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'N'}(), Val{'C'}(), A, parent(Bc), promote_type(T, V)), A, Bc)
        Base.:*(Ac::Adjoint{T,<:$Typ1{T}}, Bc::Adjoint{V,<:$Typ2{V}}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'C'}(), Val{'C'}(), parent(Ac), parent(Bc), promote_type(T, V)), Ac, Bc)

        Base.:*(At::Transpose{T,<:$Typ1{T}}, B::$Typ2{V}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'T'}(), Val{'N'}(), parent(At), B, promote_type(T, V)), At, B)
        Base.:*(A::$Typ1{T}, Bt::Transpose{V,<:$Typ2{V}}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'N'}(), Val{'T'}(), A, parent(Bt), promote_type(T, V)), A, Bt)
        Base.:*(At::Transpose{T,<:$Typ1{T}}, Bt::Transpose{V,<:$Typ2{V}}) where {T, V} = mul!(BandedMatrices.banded_similar(Val{'T'}(), Val{'T'}(), parent(At), parent(Bt), promote_type(T, V)), At, Bt)
    end
    esc(ret)
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
            BandedMatrices.rmul!(ret, -one(T))
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
        Base.:-(A::AbstractMatrix{T}, B::$Typ{V}) where {T,V} = BandedMatrices.rmul!(B - A, -1)



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
            BandedMatrices.rmul!(ret, -1)
            axpy!(1, A, ret)
        end


        LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::$Typ) = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'N'}(), A, B)



        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ{T}}, B::AbstractMatrix) where T = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:AbstractMatrix{T}}, B::$Typ) where T = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ, Bc::Adjoint{U,<:AbstractMatrix{U}}) where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'C'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, Bc::Adjoint{U,<:$Typ{U}})  where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'C'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:$Typ{T}}, Bc::Adjoint{U,<:AbstractMatrix{U}}) where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'C'}(), parent(Ac), parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix, Ac::Adjoint{T,<:AbstractMatrix{T}}, Bc::Adjoint{U,<:$Typ{U}})  where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'C'}(), Val{'C'}(), parent(Ac), parent(Bc))

        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:$Typ{T}}, B::AbstractMatrix) where T = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(At), B)
        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:AbstractMatrix{T}}, B::$Typ) where T = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(At), B)
        LinearAlgebra.mul!(C::AbstractMatrix, A::$Typ, Bt::Transpose{U,<:AbstractMatrix{U}}) where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bt))
        LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, Bt::Transpose{U,<:$Typ{U}}) where U = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bt))
        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:$Typ{T}}, Bt::Transpose{U,<:AbstractMatrix{U}}) where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(At), parent(Bt))
        LinearAlgebra.mul!(C::AbstractMatrix, At::Transpose{T,<:AbstractMatrix{T}}, Bt::Transpose{U,<:$Typ{U}}) where {T,U} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(At), parent(Bt))

        # override the mul! for real values
        LinearAlgebra.mul!(C::AbstractMatrix{T}, Ac::Adjoint{U,<:$Typ{U}}, B::AbstractMatrix{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix{T}, Ac::Adjoint{U,<:AbstractMatrix{U}}, B::$Typ{V}) where {T, U<:Real, V} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'N'}(), parent(Ac), B)
        LinearAlgebra.mul!(C::AbstractMatrix{T}, A::$Typ{U}, Bc::Adjoint{V,<:AbstractMatrix{V}}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix{T}, A::AbstractMatrix{U}, Bc::Adjoint{V,<:$Typ{V}}) where {T, U, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'N'}(), Val{'T'}(), A, parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix{T}, Ac::Adjoint{U,<:$Typ{U}}, Bc::Adjoint{V,<:AbstractMatrix{V}}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(Ac), parent(Bc))
        LinearAlgebra.mul!(C::AbstractMatrix{T}, Ac::Adjoint{U,<:AbstractMatrix{U}}, Bc::Adjoint{V,<:$Typ{V}}) where {T, U<:Real, V<:Real} = BandedMatrices.banded_matmatmul!(C, Val{'T'}(), Val{'T'}(), parent(Ac), parent(Bc))


        function Base.:*(A::$Typ{T}, B::StridedMatrix{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            mul!(Matrix{promote_type(T,V)}(undef,n,m), A, B)
        end

        function Base.:*(A::StridedMatrix{T}, B::$Typ{V}) where {T, V}
            n, m = size(A,1), size(B,2)
            mul!(Matrix{promote_type(T,V)}(undef,n,m), A, B)
        end

        Base.:*(Ac::Adjoint{T,<:$Typ{T}}, B::StridedMatrix{V}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(Ac, 1), size(B, 2)), Ac, B)
        Base.:*(Ac::Adjoint{T,<:StridedMatrix{T}}, B::$Typ{V}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(Ac, 1), size(B, 2)), Ac, B)
        Base.:*(A::$Typ{T}, Bc::Adjoint{V,<:StridedMatrix{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(A, 1), size(Bc, 2)), A, Bc)
        Base.:*(A::StridedMatrix{T}, Bc::Adjoint{V,<:$Typ{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(A, 1), size(Bc, 2)), A, Bc)
        Base.:*(Ac::Adjoint{T,<:$Typ{T}}, Bc::Adjoint{V,<:StridedMatrix{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(Ac, 1), size(Bc, 2)), Ac, Bc)
        Base.:*(Ac::Adjoint{T,<:StridedMatrix{T}}, Bc::Adjoint{V,<:$Typ{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(Ac, 1), size(Bc, 2)), Ac, Bc)

        Base.:*(At::Transpose{T,<:$Typ{T}}, B::StridedMatrix{V}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(At, 1), size(B, 2)), At, B)
        Base.:*(At::Transpose{T,<:StridedMatrix{T}}, B::$Typ{V}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(At, 1), size(B, 2)), At, B)
        Base.:*(A::$Typ{T}, Bt::Transpose{T,<:StridedMatrix{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(A, 1), size(Bt, 2)), A, Bt)
        Base.:*(A::StridedMatrix{T}, Bt::Transpose{T,<:$Typ{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(A, 1), size(Bt, 2)), A, Bt)
        Base.:*(At::Transpose{T,<:$Typ{T}}, Bt::Transpose{T,<:StridedMatrix{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(At, 1), size(Bt, 2)), At, Bt)
        Base.:*(At::Transpose{T,<:StridedMatrix{T}}, Bt::Transpose{T,<:$Typ{V}}) where {T, V} = mul!(Matrix{promote_type(T,V)}(size(At, 1), size(Bt, 2)), At, Bt)
    end
    esc(ret)
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
        BandedMatrices.MemoryLayout(A::$Typ) = BandedMatrices.BandedColumnMajor()
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
