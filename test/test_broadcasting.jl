using BandedMatrices, LinearAlgebra, ArrayLayouts, FillArrays, Test
import Base: BroadcastStyle
import Base.Broadcast: broadcasted
import BandedMatrices: BandedStyle, BandedRows

@testset "broadcasting" begin
    @testset "general" begin
        n = 1000
        for (l,u) in ((1,1), (0,0), (1,0), (0,1), (2,-1), (-1,2), (-2,2), (-2,1))
            A = brand(n,n,l,u)
            B = Matrix{Float64}(undef, n,n)
            B .= exp.(A)
            @test B == exp.(Matrix(A)) == exp.(A)

            @test exp.(A) isa BandedMatrix
            @test bandwidths(exp.(A)) == (n-1,n-1)

            C = similar(A)
            @test_throws BandError C .= exp.(A)

            @test identity.(A) == A
            @test identity.(A) isa BandedMatrix
            @test bandwidths(identity.(A)) == bandwidths(A)

            @test (z -> exp(z)-1).(A) == (z -> exp(z)-1).(Matrix(A))
            @test (z -> exp(z)-1).(A) isa BandedMatrix
            @test bandwidths((z -> exp(z)-1).(A)) == bandwidths(A)

            @test A .+ 1 == Matrix(A) .+ 1
            @test A .+ 1 isa BandedMatrix
            @test (A .+ 1) == Matrix(A) .+ 1

            @test A ./ 1 == Matrix(A) ./ 1
            @test A ./ 1 isa BandedMatrix
            @test bandwidths(A ./ 1) == bandwidths(A)

            @test 1 .+ A == 1 .+ Matrix(A)
            @test 1 .+ A isa BandedMatrix
            @test (1 .+ A) == 1 .+ Matrix(A)

            @test 1 .\ A == 1 .\ Matrix(A)
            @test 1 .\ A isa BandedMatrix
            @test bandwidths(1 .\ A) == bandwidths(A)

            A = brand(10,1,l,u)
            @test A[:,1] isa Vector
            @test norm(A .- A[:,1]) == 0
            @test A ≈ A[:,1]
        end
    end

    function test_empty(f!)
        # the function f! must not set the RHS to zero for the error check to work
        @testset "empty dest" begin
            D = brand(4, 4, -2, 1) # empty
            B = brand(size(D)...,1,2) # non-empty and non-zero bands
            @test_throws BandError f!(D, B)
            f!(D, zero(B))
            @test all(iszero, D)
        end
    end

    @testset "identity" begin
        n = 100
        A = brand(n,n,2,2)
        A.data[1,1] = NaN
        B = brand(n,n,2,2)
        B.data[1,1] = NaN
        A .= B
        @test A == B
        B = brand(n,n,1,1)
        B.data[1,1] = NaN
        A .= B
        B = brand(n,n,3,3)
        B[band(3)] .= 0
        B[band(-3)] .= 0
        B.data[1,1] = NaN
        B.data[end,end] = NaN
        A .= B
        @test A == B

        B = brand(n,n,0,3)
        B[band(3)] .= 0
        B.data[1,1] = NaN
        A .= B
        @test A == B

        B = brand(n,n,3,0)
        B[band(-3)] .= 0
        B.data[end,end] = NaN
        A .= B
        @test A == B

        B = brand(n,n,-1,1)
        A .= B
        @test A == B

        B = brand(n,n,1,-1)
        A .= B
        @test A == B

        B = brand(1,4,0,1)
        C = brand(size(B)...,0,0)
        @test_throws BandError C .= B
        @test_throws BandError C' .= B'

        @testset "empty dest" begin
            test_empty((D,B) -> D .= B)
        end

        @testset "adjtrans" begin
            @testset "empty dest" begin
                test_empty((D,B) -> D' .= B')
            end
        end
    end

    @testset "lmul!/rmul!" begin
        n = 1000
        @testset for (l,u) in ((1,1), (0,0), (1,0), (0,1), (2,-1), (-1,2), (-2,2), (-2,1))
            A = brand(n,n,l,u)
            B = brand(n,n,l+1,u+1)
            B .= (-).(A)
            @test -A isa BandedMatrix
            @test (-).(A) isa BandedMatrix
            @test bandwidths(A) == bandwidths(-A) == bandwidths((-).(A))
            @test B == -A == (-).(A)
            @test A-I isa BandedMatrix
            @test I-A isa BandedMatrix
            @test bandwidths(A-I) == bandwidths(I-A)
            @test A-I == Matrix(A) - I
            @test I-A == I - Matrix(A)
            if all(>=(0), bandwidths(A))
                @test bandwidths(A) == bandwidths(A-I)
            end
            if (l,u) == (-2,2)
                @test all(iszero, (A-I)[band(1)])
                @test all(iszero, (I-A)[band(1)])
            end

            B .= 2.0.*A
            @test B ==  2A == 2.0.*A
            @test 2A isa BandedMatrix
            @test 2.0.*A isa BandedMatrix
            @test bandwidths(2A) == bandwidths(2.0.*A) == bandwidths(A)

            A .= 2.0.*A
            @test A == B

            B .= A.*2.0
            @test B ==  A*2 == A.*2.0
            @test A*2 isa BandedMatrix
            @test A .* 2.0 isa BandedMatrix
            @test bandwidths(A*2) == bandwidths(A.*2.0) == bandwidths(A)
            A .= A.*2.0
            @test A == B

            B .= A ./ 2.0
            @test B == A/2 == A ./ 2.0
            @test A/2 isa BandedMatrix
            @test A ./ 2.0 isa BandedMatrix
            @test bandwidths(A/2) == bandwidths(A ./ 2.0) == bandwidths(A)

            B .= 2.0 .\ A
            @test B == A/2 == A ./ 2.0
            @test 2\A isa BandedMatrix
            @test 2.0 .\ A isa BandedMatrix
            @test bandwidths(2\A) == bandwidths(2.0 .\ A) == bandwidths(A)

            if -l <= u
                # This doesn't work for now if there are no bands,
                # although ideally, the matrix should be all NaN
                A.data .= NaN
                lmul!(0.0,A)
                @test isnan(norm(A)) == isnan(norm(lmul!(0.0,[NaN])))
                lmul!(false,A)
                @test norm(A) == 0.0

                A.data .= NaN
                rmul!(A,0.0)
                @test isnan(norm(A)) == isnan(norm(rmul!([NaN],0.0)))
                rmul!(A,false)
                @test norm(A) == 0.0
            end
        end

        n = 100
        A = brand(n,n,2,2)
        B = brand(n,n,1,1)
        A[band(2)] .= A[band(-2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)

        n = 100
        A = brand(n,n,2,2)
        B = brand(n,n,1,3)
        A[band(-2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)

        A = brand(n,n,2,2)
        B = brand(n,n,3,1)
        A[band(2)] .= 0
        B .= A ./ 2.0
        @test B == A / 2.0 == Matrix(A)/2.0

        B .= 2.0 .\ A
        @test B == 2.0 \ A == 2.0 \ Matrix(A)

        @testset "empty dest" begin
            test_empty((D,B) -> D .= 2 .* B)
            test_empty((D,B) -> D .= B .* 2)
        end

        @testset "trans-adj" begin
            A = brand(5,5,1,1)
            Ã = copy(A)

            lmul!(2.0, A')
            @test A == 2Ã
            lmul!(2.0, transpose(A))
            @test A == 4Ã
            lmul!(2.0, Symmetric(A))
            @test A == 8Ã
            lmul!(2.0, Hermitian(A))
            @test A == 16Ã
            rmul!(Hermitian(A), 1/2)
            @test A == 8Ã
            rmul!(Symmetric(A), 1/2)
            @test A == 4Ã
            rmul!(transpose(A), 1/2)
            @test A == 2Ã
            rmul!(A', 1/2)
            @test A == Ã

            @testset "empty dest" begin
                test_empty((D,B) -> D' .= 2 .* B')
                test_empty((D,B) -> D' .= B' .* 2)
                test_empty((D,B) -> D' .= 2 .* B' .* 2)
                test_empty((D,B) -> D' .= 2 .* B)
                test_empty((D,B) -> D' .= B .* 2)
                test_empty((D,B) -> D' .= 2 .* B .* 2)
            end
        end
    end

    @testset "axpy!" begin
        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        C .= A .+ B
        @test C == A + B == A .+ B  == Matrix(A) + Matrix(B)
        @test A + B isa BandedMatrix
        @test A .+ B isa BandedMatrix
        @test bandwidths(A+B) == bandwidths(A.+B) == (2,2)
        B .= A .+ B
        @test B == C

        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        C .= A .* B
        @test C == A .* B  == Matrix(A) .* Matrix(B)
        @test A .* B isa BandedMatrix
        @test bandwidths(A.*B) == (1,1)
        @time B .= A .* B
        @test B == C

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(2)] .= B[band(-2)] .= 0
        C = brand(n,n,1,1)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(-2)] .= 0
        C = brand(n,n,1,2)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        B[band(2)] .= 0
        C = brand(n,n,2,1)
        C .= A .+ B
        @test C == A + B == A .+ B == Matrix(A) + Matrix(B)

        n = 100
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)

        C .= 2.0 .* A .+ B
        @test C == 2A+B == 2.0.*A .+ B
        @test all(axpy!(2.0, A, copy(B)) .=== C)

        @test 2A + B isa BandedMatrix
        @test 2.0.*A .+ B isa BandedMatrix
        @test bandwidths(2A+B) == bandwidths(2.0.*A .+ B) == (2,2)
        B .= 2.0 .* A .+ B
        @test B == C

        # test with identical bandwidth
        @test axpy!(3, A, copy(A)) ≈ 4A

        @testset "trivial cases" begin
            B = brand(2,4,-1,0) # no bands in B
            B2 = brand(2,4,0,-1) # no bands in B2
            C = brand(size(B)...,1,1)
            D = copy(C)
            axpy!(0.1, B, C) # no bands in src
            @test C == D
            @test_throws BandError axpy!(0.1, C, B)
            @test_throws BandError axpy!(0.1, C, B2)
            D = copy(B)
            C .= 0
            axpy!(0.1, C, B) # no bands in dest, but src is zero
            @test B == D
        end
    end

    @testset "gbmv!" begin
        n = 100
        x = randn(n)
        y = similar(x)
        z = similar(y)
        @testset for (l,u) in ((1,1), (-1,1), (1,-1), (-2,1), (0,1), (1,0), (0,0))
            A = brand(n,n,l,u)
            mul!(y,A,x)
            @test Matrix(A)*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('N', n, l, u, 1.0, A.data, x, 0.0, copy(y)) .===
                        Broadcast.materialize!(MulAdd(1.0,A,x,0.0,similar(y))) .=== y)
            end

            z .= MulAdd(2.0,A,x,3.0,y)
            @test 2Matrix(A)*x + 3y ≈ z
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('N', n, l, u, 2.0, A.data, x, 3.0, copy(y)) .=== z)
            end

            @test MemoryLayout(typeof(A')) == BandedRows{DenseColumnMajor}()

            mul!(y,A',x)
            @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('T', n, l, u, 1.0, A.data, x, 0.0, copy(y)) .=== y)
            end

            z .= MulAdd(2.0,A',x,3.0,y)
            @test 2Matrix(A')*x + 3y ≈ z
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('T', n, l, u, 2.0, A.data, x, 3.0, copy(y)) .=== z)
            end

            @test MemoryLayout(typeof(transpose(A))) == BandedRows{DenseColumnMajor}()

            muladd!(1.0,transpose(A),x,0.0,y)
            @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('T', n, l, u, 1.0, A.data, x, 0.0, copy(y)) .=== y)
            end
        end

        n = 100
        x = randn(ComplexF64,n)
        y = similar(x)
        yc = similar(x)
        z = similar(y)
        @testset for (l,u) in ((1,1), (-1,1), (1,-1), (-2,1), (0,1), (1,0), (0,0))
            A = brand(ComplexF64,n,n,l,u)

            mul!(y,A,x)
            @test Matrix(A)*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('N', n, l, u, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
            end

            @test MemoryLayout(typeof(A')) == ConjLayout{BandedRows{DenseColumnMajor}}()

            z .= MulAdd(2.0+0.0im,A,x,3.0+0.0im,y)
            @test 2Matrix(A)*x + 3y ≈ z
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('N', n, l, u, 2.0+0.0im, A.data, x, 3.0+0.0im, copy(y)) .=== z)
            end

            mul!(y,A',x)
            @test Matrix(A')*x ≈ Matrix(A)'*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('C', n, l, u, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
            end

            @test MemoryLayout(typeof(transpose(A))) == BandedRows{DenseColumnMajor}()

            mul!(y,transpose(A),x)
            @test Matrix(transpose(A))*x ≈ transpose(Matrix(A))*x ≈ y
            if l >= 0 && u >= 0
                @test all(BLAS.gbmv!('T', n, l, u, 1.0+0.0im, A.data, x, 0.0+0.0im, copy(y)) .=== y)
            end
        end
    end

    @testset "non-blas matvec" begin
        n = 10
        A = brand(Int,n,n,1,1)
        x = rand(Int,n)
        y = similar(x)
        mul!(y,A,x)
        @test y == Matrix(A)*x
        mul!(y,A',x)
        @test y == Matrix(A')*x
        mul!(y,transpose(A),x)
        @test y == Matrix(transpose(A))*x

        A = brand(Int,n,n,1,1) + im*brand(Int,n,n,1,1)
        x = rand(eltype(A),n)
        y = similar(x)
        mul!(y,A,x)
        @test y == Matrix(A)*x
        mul!(y,A',x)
        @test y == Matrix(A')*x
        mul!(y,transpose(A),x)
        @test y == Matrix(transpose(A))*x
    end

    @testset "gbmm!" begin
        n = 10
        A = brand(n,n,1,1)
        B = brand(n,n,2,2)
        C = brand(n,n,3,3)
        mul!(C,A,B)
        @test Matrix(C) ≈ Matrix(A)*Matrix(B)
        mul!(C,A',B)
        @test Matrix(C) ≈ Matrix(A')*Matrix(B)
        mul!(C,A,B')
        @test Matrix(C) ≈ Matrix(A)*Matrix(B')
        C = brand(n,n,4,4)
        mul!(C,A,B)
        @test Matrix(C) ≈ Matrix(A)*Matrix(B)
        B = randn(n,n)
        C = similar(B)

        mul!(C,A,B)
        @test C ≈ Matrix(A)*Matrix(B)
        mul!(C,B,A)
        @test C ≈ Matrix(B)*Matrix(A)
    end

    @testset "Subarray" begin
        n = 10
        A = brand(n,n,1,1)
        B = view(brand(2n,2n,2,0),Base.OneTo(n),1:n)
        @test BroadcastStyle(typeof(B)) == BandedStyle()

        @test A .+ B isa BandedMatrix
        @test A + B isa BandedMatrix
        @test bandwidths(A .+ B) == (2,1)
        @test A .+ B == Matrix(A) + Matrix(B)
    end

    @testset "adjtrans" begin
        n = 10
        A = brand(n,n,1,1)

        @test BroadcastStyle(typeof(A')) isa BandedStyle
        @test BroadcastStyle(typeof(transpose(A))) isa BandedStyle

        @test A' .+ A isa BandedMatrix
        @test transpose(A) .+ A isa BandedMatrix
        @test A' .+ A == transpose(A) .+ A == Matrix(A)' .+ A
    end

    @testset "vector and matrix broadcastring" begin
        n = 10
        A = brand(n,n,1,2)
        b = rand(n) .+ 0.1 # avoid zero
        Bb = BandedMatrix(b')'
        Bb2 = BandedMatrix(b', (0,3))'

        @testset "implicit dest" begin
            for A_ in (A, A'), b_ in (b, Bb, Bb2)
                @test b_ .* A_ == b_ .* Matrix(A_)
                @test b_ .* A_ isa BandedMatrix
                @test bandwidths(b_ .* A_) == bandwidths(A_)

                @test b_' .* A_ == b_' .* Matrix(A_)
                @test b_' .* A_ isa BandedMatrix
                @test bandwidths(b_' .* A_) == bandwidths(A_)

                @test permutedims(b_) .* A_ == permutedims(b_) .* Matrix(A_)
                @test permutedims(b_) .* A_ isa BandedMatrix
                @test bandwidths(permutedims(b_) .* A_) == bandwidths(A_)

                @test A_ .* b_ == Matrix(A_) .* b_
                @test A_ .* b_ isa BandedMatrix
                @test bandwidths(A_ .* b_) == bandwidths(A_)

                @test A_ .* b_' == Matrix(A_) .* b_'
                @test A_ .* b_' isa BandedMatrix
                @test bandwidths(A_ .* b_') == bandwidths(A_)

                @test A_ .* permutedims(b_) == Matrix(A_) .* permutedims(b_)
                @test A_ .* permutedims(b_) isa BandedMatrix
                @test bandwidths(A_ .* permutedims(b_)) == bandwidths(A_)
            end

            # division tests currently don't deal with Inf/NaN correctly,
            # so we don't divide by zero
            for A_ in (A, A'), b_ in (b, Bb)
                @test b_ .\ A_ == b_ .\ Matrix(A_)
                @test b_ .\ A_ isa BandedMatrix
                @test bandwidths(b_ .\ A_) == bandwidths(A_)
                @test A_ ./ b_ == Matrix(A_) ./ b_
                @test A_ ./ b_ isa BandedMatrix
                @test bandwidths(A_ ./ b_) == bandwidths(A_)
            end

            @test bandwidths(Broadcast.broadcasted(/, b, A)) == (9,9)
            @test isinf((b ./ A)[4,1])
            @test bandwidths(Broadcast.broadcasted(\, A,b)) == (9,9)
            @test isinf((A .\ b)[4,1])

            @test reshape(b,10,1) .* A isa BandedMatrix
            @test reshape(b,10,1) .* A == A .* reshape(b,10,1) == A .* b
            @test bandwidths(reshape(b,10,1) .* A) == bandwidths(A)
            @test bandwidths(A .* reshape(b,10,1)) == bandwidths(A)

            @test b .+ A == b .+ Matrix(A) == A .+ b

            @test bandwidths(broadcasted(+, A, b')) == (9,9)
            @test A .+ b' == b' .+ A == Matrix(A) .+ b'
            @test bandwidths(A .+ b') == bandwidths(b' .+ A)  == (9,9)

            @testset "nested broadcast" begin
                @test bandwidths((b ./ 2) .* A) == (1,2)
                @test (b ./ 2) .* A == (b ./ 2) .* Matrix(A)
            end
        end

        @testset "explicit dest" begin
            D = brand(size(A)..., (bandwidths(A) .+ 1)...)
            D .= 0

            for (A_, D_) in ((A,D), (A', D')), b_ in (b, Bb, Bb2)
                D_ .= b_ .* A_
                @test D_ == b_ .* A_

                D_ .= b_' .* A_
                @test D_ == b_' .* A_

                D_ .= permutedims(b_) .* A_
                @test D_ == permutedims(b_) .* A_

                D_ .= A_ .* b_
                @test D_ == A_ .* b_

                D_ .= A_ .* b_'
                @test D_ == A_ .* b_'

                D_ .= A_ .* permutedims(b_)
                @test D_ == A_ .* permutedims(b_)
            end
        end
    end

    @testset "views" begin
        A = BandedMatrix(Ones(10,10),(1,1))
        B = 2A
        view(A,8:10,:) .= view(B,8:10,:)
        @test A[1,1] == 1.0
        @test A[6,7] == 1.0
        @test A[7,7] == 1.0
        @test A[8,7] == 2.0

        A = BandedMatrix{Int}(undef,(10,10),(2,2));  vec(A.data) .= (1:length(A.data));
        B = BandedMatrix{Int}(undef,(10,10),(1,1));  vec(B.data) .= (1:length(B.data));
        Ã = Matrix(A); B̃ = Matrix(B)
        view(A,5:10,:) .= view(B,5:10,:)
        view(Ã,5:10,:) .= view(B̃,5:10,:)
        @test Ã == A

        A = BandedMatrix{Int}(undef,(10,10),(2,1));  vec(A.data) .= (1:length(A.data));
        B = BandedMatrix{Int}(undef,(10,10),(1,2));  vec(B.data) .= (1:length(B.data));
        @test_throws BandError view(A,5:10,:) .= view(B,5:10,:)

        A = BandedMatrix{Int}(undef,(10,10),(1,2));  vec(A.data) .= (1:length(A.data));
        B = BandedMatrix{Int}(undef,(10,10),(2,1));  vec(B.data) .= (1:length(B.data));
        @test_throws BandError view(A,5:10,:) .= view(B,5:10,:)

        A = BandedMatrix{Int}(undef,(10,10),(1,2));  vec(A.data) .= (1:length(A.data));
        B = BandedMatrix{Int}(undef,(10,10),(2,1));  vec(B.data) .= (1:length(B.data));
        view(A,3:4,2:6) .= view(B,8:9,1:5)
        @test A[3:4,2:6] == zeros(Int,2,5)
        @test A[5,4] == 16

        A = BandedMatrix{Int}(undef,(10,10),(1,2));  vec(A.data) .= (1:length(A.data));
        B = BandedMatrix{Int}(undef,(10,10),(2,1));  vec(B.data) .= (1:length(B.data));
        view(A,3:4,2:6) .= view(B,1:2,4:8)
        @test A[3:4,:] == zeros(Int,2,10)

        A = BandedMatrix{Int}(undef,(1,1),(1,1)); vec(A.data) .= 2*(1:length(A.data));
        B = BandedMatrix{Int}(undef,(1,1),(1,1)); vec(B.data) .= (1:length(B.data));
        view(A,2:1,2:1) .= view(B,2:1,2:1)
        @test A[1,1] == 4
    end

    @testset "special broasdcast" begin
        A = brand(5,5,1,1)
        b = brand(5,1,1,1)
        B = brand(1,5,1,1)
        @test A .+ b == b .+ A == Matrix(A) .+ Matrix(b)
        @test A .+ B' == B' .+ A == Matrix(A) .+ Matrix(B)'
        @test A .+ b' == b' .+ A == Matrix(A) .+ Matrix(b)'
        @test A .+ B == B .+ A == Matrix(A) .+ Matrix(B)

        @test bandwidths(A .* b) == bandwidths(A .* b') == bandwidths(b .* A) == bandwidths(b' .* A) == (1,1)
        @test A .* b == b .* A == Matrix(A) .* Matrix(b)
        @test A .* b' == b' .* A == Matrix(A) .* Matrix(b')
        @test A .* B == B .* A == Matrix(A) .* Matrix(B)
    end

    @testset "broadcast with Int (#154)" begin
        b = BandedMatrix(8Ones(10,10), (2,1))
        @test BandedMatrices._isweakzero(round,Int,b)
        @test round.(Int, b) == round.(Int,Matrix(b))
    end

    @testset "degnerate bands" begin
		A = BandedMatrix(Zeros(1,1), (0,-3))
		C = BandedMatrix(Ones(1,1), (0,0))
		C .= 0.0 .- A
		@test C == zeros(1,1)
		C .= A .- 0.0
        @test C == zeros(1,1)

        A = BandedMatrix(Eye(3,4))
        B = randn(3,4)
        bc = broadcasted(*, A, B)

        @test copyto!(similar(bc, Float64), bc) == A .* B == B .* A ==
            BandedMatrix(B, (0,0)) == Matrix(A) .* B
    end

    @testset "adding degenerate" begin
        A = BandedMatrix(1 => 1:9)
        B = BandedMatrix(-1 => 1:9)
        C = BandedMatrix(Fill(-1,10,10),(1,1))
        C .= A .+ B
        @test diag(C) == Zeros{Int}(10)
        C = BandedMatrix(Fill(-1,10,10),(1,1))
        C .= A .+ B
        @test diag(C) == Zeros{Int}(10)
    end

    @testset "nested broadcast" begin
        A = brand(5,4,2,1)
        x = randn(5)
        B = Base.Broadcast.broadcasted(*, Base.Broadcast.broadcasted(+, 2, x), A)
        @test bandwidths(B) == bandwidths(A) == bandwidths(materialize(B)) == (2,1)
        @test materialize(B) == (2 .+ x) .* Matrix(A)

        B = Base.Broadcast.broadcasted(+, Base.Broadcast.broadcasted(+, A, A), A)
        @test bandwidths(B) == bandwidths(A)
        @test (A .+ A) .+ A == 3A
    end

    @testset "ones special case" begin
        A = brand(5,4,2,1)
        @test Ones(5) .* A == A
        @test Ones(5,4) .* A == A
        @test Ones(5) .\ A == A
        @test Ones(5,4) .\ A == A
        @test Ones(5) ./ A == inv.(A)
        @test_throws DimensionMismatch Ones(3) .* A
        @test_throws DimensionMismatch Ones(3,4) .* A

        @test A .* Ones(5) == A
        @test A .* Ones(5,4) == A
        @test A ./ Ones(5) == A
        @test A ./ Ones(5,4) == A
        @test A .\ Ones(5) == inv.(A)
        @test_throws DimensionMismatch A .* Ones(3)
        @test_throws DimensionMismatch A .* Ones(3,4)
    end

    @testset "degenerate bands" begin
        A = BandedMatrix{Float64}(undef, (5, 5), (1,-1)); A.data .= NaN
        B = BandedMatrix{Float64}(undef, (5, 5), (-1,1)); B.data .= NaN
        Z = Diagonal(Zeros(5))
        copyto!(A, Z)
        copyto!(B, Z)
        @test A == B == zeros(5,5)
    end
end
