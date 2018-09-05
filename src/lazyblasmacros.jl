# These are temporary to work around memory bug. They avoid use of <:

macro _blasmatvec(Lay, Typ)
    esc(quote
        # y .= Mul(A,b) gets lowered here
        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                                             M::LazyArrays.MatMulVec{T, $Lay, <:LazyArrays.AbstractStridedLayout, T, T}) where T<: $Typ
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, one(T), zero(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVec{T, $Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            α,M = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, α, zero(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,y = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), one(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusConstVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,βc = bc.args
            β,y = βc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), β)
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,y = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, α, one(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusConstVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            β,y = βc.args
            LazyArrays.blasmul!(dest, A, x, y, α, β)
        end
    end)
end

macro blasmatvec(Lay)
    esc(quote
        LazyArrays.@_blasmatvec $Lay LinearAlgebra.BLAS.BlasFloat
        LazyArrays.@_blasmatvec LazyArrays.ConjLayout{$Lay} LinearAlgebra.BLAS.BlasComplex
    end)
end


macro _blasmatmat(CTyp, ATyp, BTyp, Typ)
    esc(quote
        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 M::LazyArrays.MatMulMat{T,$ATyp,$BTyp,T,T}) where T<: $Typ
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, one(T), zero(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMat{T,$ATyp,$BTyp}) where T<: $Typ
            α,M = bc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, α, zero(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusMat{T,$ATyp,$BTyp}) where T<: $Typ
            M,C = bc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, one(T), one(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusConstMat{T,$ATyp,$BTyp}) where T<: $Typ
            M,βc = bc.args
            β,C = βc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, one(T), β)
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusMat{T,$ATyp,$BTyp}) where T<: $Typ
            αM,C = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, α, one(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusConstMat{T,$ATyp,$BTyp}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            β,C = βc.args
            LazyArrays.blasmul!(dest, A, B, C, α, β)
        end
    end)
end


macro blasmatmat(ATyp, BTyp, CTyp)
    esc(quote
        LazyArrays.@_blasmatmat $ATyp $BTyp $CTyp BlasFloat
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} $BTyp $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} LazyArrays.ConjLayout{$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp $BTyp LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} $BTyp LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{$BTyp} LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} LazyArrays.ConjLayout{$BTyp} LazyArrays.ConjLayout{$CTyp} BlasComplex
    end)
end
