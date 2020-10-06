using BandedMatrices, LinearAlgebra, FillArrays, Test
import BandedMatrices: _BandedMatrix

@testset "dot(x, A::BandedMatrix, y)" begin
    @testset "Single, constant diagonal" begin
        for (m,n,l) in [(100,15,5),(15,100,-5)]
            S = _BandedMatrix(0.1Ones{Int}(1,n), Base.OneTo(m), l, -l)
            MS = Matrix(S)

            a = rand(ComplexF64, m)
            b = rand(ComplexF64, n)

            v = dot(a, MS, b)
            ref = dot(a, S, b)
            @test v ≈ ref rtol=1e-14
        end
    end

    @testset "Multiple, random diagonals" begin
        for cols in [2000,500]
            for (l,u) in [(3,-1), (1,1), (-1,3)]
                S = _BandedMatrix(rand(3,cols), Base.OneTo(1000), l, u)
                MS = Matrix(S)
                m,n = size(S)

                a = rand(ComplexF64, m)
                b = rand(ComplexF64, n)

                v = dot(a, S, b)
                ref = dot(a, MS, b)

                @test v ≈ ref rtol=1e-14
            end
        end
    end
end
