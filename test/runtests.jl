versioninfo()

using BandedMatrices, Base.Test

A,B=brand(10,12,2,3),brand(10,12,3,4)


@test_approx_eq full(sparse(A)) full(A)

@test_approx_eq full(A') full(A)'
@test_approx_eq full(A.') full(A).'
@test_approx_eq full((A+im*A)') (full(A)+im*full(A))'
@test_approx_eq full((A+im*A).') (full(A)+im*full(A)).'

@test_approx_eq full(A+B) (full(A)+full(B))
@test_approx_eq full(A-B) (full(A)-full(B))

@test_approx_eq full(A.*B) (full(A).*full(B))

C,D=brand(10,10,2,3),brand(12,12,3,4)

@test_approx_eq full(C*A) full(C)*full(A)
@test_approx_eq full(A*D) full(A)*full(D)


v=rand(12)
w=rand(10)

@test_approx_eq A*v full(A)*v
@test_approx_eq A'*w full(A)'*w




A=brand(10000,10000,2,3)
B=brand(1000,1000,200,300)
v=rand(10000)
w=rand(1000)

A*v
@time A*v
println("Time should be   0.000133 seconds (148 allocations: 88.183 KB)")
@time B*w
println("Time should be   0.000799 seconds (5 allocations: 8.047 KB)")



show(brand(10,10,3,3))