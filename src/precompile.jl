using PrecompileTools

@setup_workload begin
	vs = ([1.0], Float32[1.0], ComplexF32[1.0], ComplexF64[1.0])
	Bs = Any[BandedMatrix(0 => v) for v in vs]
	@compile_workload begin
		for B in Bs, op in (+, -, *)
			op(B, B)
		end
		for (v, B) in zip(vs, Bs)
			B * v
		end
	end
end
