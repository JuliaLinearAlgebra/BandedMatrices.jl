using BandedMatrices, FillArrays, ArrayLayouts, Test
import LinearAlgebra: axpy!
import BandedMatrices: BandedColumns, bandeddata, MemoryLayout, BandedSubBandedMatrix, _BandedMatrix, sub_materialize


@testset "BandedMatrix SubArray" begin
    @testset "BandedMatrix SubArray interface" begin
        A = brand(10,10,1,2)
        V = view(A,2:4,3:6)
        @test isbanded(V)
        @test bandwidths(V) == (2,1)
        @test MemoryLayout(V) isa BandedColumns{DenseColumnMajor}
        @test all(Matrix(_BandedMatrix(bandeddata(V), size(V,1), bandwidths(V)...)) .===
                Matrix(V))

        @test sub_materialize(V) isa BandedMatrix
        @test sub_materialize(MemoryLayout(V), V, (Base.Slice(1:3), Base.OneTo(4))) isa BandedMatrix
        # if 2nd argument is not Base.OneTo, cannot convert to BandedMatrix so take adjoint
        @test sub_materialize(MemoryLayout(V), V, (Base.OneTo(3), Base.Slice(1:4))) isa Adjoint{<:Any,<:BandedMatrix}
        # if neither argument is Base.OneTo, we give up
        @test sub_materialize(MemoryLayout(V), V, (Base.Slice(1:3), Base.Slice(1:4))) isa Matrix
    end

    @testset "BandedMatrix SubArray arithmetic" begin
        let A = brand(10,10,1,2), B = brand(20,20,1,2)
            @test isa(view(B,1:10,1:10),BandedSubBandedMatrix{Float64})
            @test MemoryLayout(typeof(view(B,1:10,1:10))) isa BandedColumns{DenseColumnMajor}

            B2 = copy(B)
            @test B == B2
            @test (2.0A+B[1:10,1:10]) ≈ axpy!(2.0,A,view(B2,1:10,1:10))
            @test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

            A2 = copy(A)
            @test (2.0B[1:10,1:10]+A) ≈ axpy!(2.0,view(B,1:10,1:10),A2)
            @test (2.0B[1:10,1:10]+A) ≈ A2
        end

        let A = brand(20,20,1,2), B = brand(20,20,1,2)
            @test isa(view(B,1:10,1:10),BandedSubBandedMatrix{Float64})

            B2 = copy(B)
            @test (2.0A[1:10,1:10]+B[1:10,1:10]) ≈ axpy!(2.0,view(A,1:10,1:10),view(B2,1:10,1:10))
            @test (2.0A[1:10,1:10]+B[1:10,1:10]) ≈ B2[1:10,1:10]

            B2 = copy(B)
            @test (2.0A[1:10,:]+B[1:10,:]) ≈ axpy!(2.0,view(A,1:10,:),view(B2,1:10,:))
            @test (2.0A[1:10,:]+B[1:10,:]) ≈ B2[1:10,:]

            B2 = copy(B)
            @test (2.0A[:,1:10]+B[:,1:10]) ≈ axpy!(2.0,view(A,:,1:10),view(B2,:,1:10))
            @test (2.0A[:,1:10]+B[:,1:10]) ≈ B2[:,1:10]


            B2 = copy(B)
            @test (2.0A[:,:]+B[:,:]) ≈ axpy!(2.0,view(A,:,:),view(B2,:,:))
            @test (2.0A[:,:]+B[:,:]) ≈ B2[:,:]
        end

        let A = brand(10,10,1,2), B = brand(20,10,1,2)
            B2 = copy(B)
            @test (2.0A+B[1:10,1:10]) ≈ axpy!(2.0,A,view(B2,1:10,:))
            @test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

            A2 = copy(A)
            @test (2.0B[1:10,1:10]+A) ≈ axpy!(2.0,view(B,1:10,:),A2)
            @test (2.0B[1:10,1:10]+A) ≈ A2
        end

        let A = brand(10,10,1,2), B = brand(10,20,1,2)
            B2 = copy(B)
            @test (2.0A+B[1:10,1:10]) ≈ axpy!(2.0,A,view(B2,:,1:10))
            @test (2.0A+B[1:10,1:10]) ≈ B2[1:10,1:10]

            A2 = copy(A)
            @test (2.0B[1:10,1:10]+A) ≈ axpy!(2.0,view(B,:,1:10),A2)
            @test (2.0B[1:10,1:10]+A) ≈ A2
        end
    end

    @testset "BandedMatrix SubArray *" begin
        let A=brand(10,10,1,2), v=rand(20)
            @test A*view(v,1:10) ≈ Matrix(A)*v[1:10]
            @test A*view(v,1:2:20) ≈ Matrix(A)*v[1:2:20]

            M=rand(20,20)
            @test A*view(M,1:10,1:10) ≈ Matrix(A)*M[1:10,1:10]
            @test A*view(M,1:2:20,1:2:20) ≈ Matrix(A)*M[1:2:20,1:2:20]
        end

        let A=brand(10,10,1,2), B=brand(20,10,1,2), C=brand(10,20,1,2),
            D=brand(20,20,1,2), M=rand(10,10), V=view(rand(20,20),1:2:20,1:2:20)


            for S in (view(A,:,:),view(B,1:10,:),view(C,:,1:10),view(D,1:10,1:10),
                        view(D,2:11,1:10),view(D,1:10,2:11),view(D,11:20,11:20))
                @test A*S ≈ Matrix(A)*Matrix(S)
                @test S*A ≈ Matrix(S)*Matrix(A)
                @test S*S ≈ Matrix(S)*Matrix(S)

                @test M*S ≈ M*Matrix(S)
                @test S*M ≈ Matrix(S)*M

                @test V*S ≈ Matrix(V)*Matrix(S)
                @test S*V ≈ Matrix(S)*Matrix(V)
            end
        end
    end

    @testset "BandedMatrix SubArray conversion" begin
        @testset "Container $C" for C in (Matrix{Int32}, MyMatrix{Int32})
            matrix = convert(C, rand(Int32, 10, 12))
            T = eltype(matrix)

            banded = @inferred BandedMatrix{T, C}(matrix, (1, 1))
            indices = 2:8, 3:9, 5:9
            @testset "view $kr, $jr" for kr in indices, jr in indices
                bview = @inferred view(banded, kr, jr)
                @test bview isa BandedMatrices.BandedSubBandedMatrix{T, C}
                @test @inferred(convert(BandedMatrix, bview)) isa BandedMatrix{T, C}
                @test convert(BandedMatrix, bview) == banded[kr, jr]
            end
        end
    end
end
