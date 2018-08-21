


macro banded(Typ)
    ret = quote
        BandedMatrices.MemoryLayout(A::$Typ) = BandedMatrices.BandedColumnMajor()
        BandedMatrices.isbanded(::$Typ) = true
    end
    esc(ret)
end
