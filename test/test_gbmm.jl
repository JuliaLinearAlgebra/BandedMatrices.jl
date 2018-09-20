using BandedMatrices, LazyArrays, Test

@testset "gbmm!" begin
    # test gbmm! subpieces step by step and column by column
    for n in (1,5,50), ν in (1,5,50), m in (1,5,50),
                    Al in (0,1,2,30), Au in (0,1,2,30),
                    Bl in (0,1,2,30), Bu in (0,1,2,30)
        A=brand(n,ν,Al,Au)
        B=brand(ν,m,Bl,Bu)
        α,β,T=0.123,0.456,Float64
        C=brand(Float64,n,m,A.l+B.l,A.u+B.u)
        a=pointer(A.data)
        b=pointer(B.data)
        c=pointer(C.data)
        sta=max(1,stride(A.data,2))
        stb=max(1,stride(B.data,2))
        stc=max(1,stride(C.data,2))

        sz=sizeof(T)

        mr=1:min(m,1+B.u)
        exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
        for j=mr
            BandedMatrices.A11_Btop_Ctop_gbmv!(α,β,
                                           n,ν,m,j,
                                           sz,
                                           a,A.l,A.u,sta,
                                           b,B.l,B.u,stb,
                                           c,C.l,C.u,stc)
       end
        @test C[:,mr] ≈ exC[:,mr]

        mr=1+B.u:min(1+C.u,ν+B.u,m)
        exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
        for j=mr
            BandedMatrices.Atop_Bmid_Ctop_gbmv!(α,β,
                                           n,ν,m,j,
                                           sz,
                                           a,A.l,A.u,sta,
                                           b,B.l,B.u,stb,
                                           c,C.l,C.u,stc)
       end
       if !isempty(mr)
           @test C[:,mr] ≈ exC[:,mr]
       end

       mr=1+C.u:min(m,ν+B.u,n+C.u)
       exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
       for j=mr
           BandedMatrices.Amid_Bmid_Cmid_gbmv!(α,β,
                                          n,ν,m,j,
                                          sz,
                                          a,A.l,A.u,sta,
                                          b,B.l,B.u,stb,
                                          c,C.l,C.u,stc)
      end
      if !isempty(mr)
          @test C[:,mr] ≈ exC[:,mr]
      end

      mr=ν+B.u+1:min(m,n+C.u)
      exC=(β*Matrix(C)+α*Matrix(A)*Matrix(B))
      for j=mr
          BandedMatrices.Anon_Bnon_C_gbmv!(α,β,
                                         n,ν,m,j,
                                         sz,
                                         a,A.l,A.u,sta,
                                         b,B.l,B.u,stb,
                                         c,C.l,C.u,stc)
     end
     if !isempty(mr)
         @test C[:,mr] ≈ exC[:,mr]
     end
    end


    # test gbmm!


    for n in (1,5,50), ν in (1,5,50), m in (1,5,50), Al in (0,1,2,30), Au in (0,1,2,30), Bl in (0,1,2,30), Bu in (0,1,2,30)
        A=brand(n,ν,Al,Au)
        B=brand(ν,m,Bl,Bu)
        α,β,T=0.123,0.456,Float64
        C=brand(Float64,n,m,A.l+B.l,A.u+B.u)

        exC=α*Matrix(A)*Matrix(B)+β*Matrix(C)
        BandedMatrices.gbmm!('N','N', α,A,B,β,C)

        @test Matrix(exC) ≈ Matrix(C)
    end
end


@testset "Negative bands fills with zero" begin
    A = brand(10,10,2,2)
    B = brand(10,10,-2,2)
    C = BandedMatrix(Fill(NaN,10,10),(0,4))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)


    A = brand(10,10,-2,2)
    B = brand(10,10,-2,2)
    C = BandedMatrix(Fill(NaN,10,10),(-4,4))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)

    A = brand(10,10,-2,2)
    B = brand(10,10,2,2)
    C = BandedMatrix(Fill(NaN,10,10),(0,4))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)

    A = brand(10,10,2,2)
    B = brand(10,10,2,-2)
    C = BandedMatrix(Fill(NaN,10,10),(4,0))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)


    A = brand(10,10,2,-2)
    B = brand(10,10,2,-2)
    C = BandedMatrix(Fill(NaN,10,10),(4,-4))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)

    A = brand(10,10,2,-2)
    B = brand(10,10,2,2)
    C = BandedMatrix(Fill(NaN,10,10),(4,0))
    C .= Mul(A,B)
    @test C == Matrix(A)*Matrix(B)

    A = brand(30,1,0,0)
    B = brand(1,30,17,17)
    C = BandedMatrix(Fill(NaN, 30,30), (17,17))
    C .= Mul(A,B)
    @test C == A*B
end

@testset "Not enough bands" begin
    A = BandedMatrix(Zeros(10,10), (1,1))
    A[band(0)] .= randn(10)
    B = BandedMatrix(randn(10,10), (1,1))
    C = BandedMatrix(Zeros(10,10), (1,1))

    C .= Mul(A,B)

    @test all(C .=== A*B)

    A[band(1)] .= randn(9)
    @test_throws BandError C .= Mul(A,B)
end


@testset "BandedMatrix{Int} * Vector{Vector{Int}}" begin
    A, x =  [1 2; 3 4] , [[1,2],[3,4]]
    @test BandedMatrix(A)*x == A*x
end
