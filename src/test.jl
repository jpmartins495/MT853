using SparseArrays, MAT, LinearAlgebra

test_dict = matread("$(pwd())\\src\\coord-examples\\SC11_alt.mat")

A, ftarget, b, γ, L, Lₘₐₓ = getindex.(Ref(test_dict), ["A", "ftarget", "b", "gamma", "L", "Lmax"])
b = vec(b) # Transforma b de Array{, 2}(n,1) para Vector{}(n,)

@info "Variáveis do teste:" A=summary(A) ftarget=ftarget b=summary(b) γ=γ L=L Lₘₐₓ=Lₘₐₓ

f(x:: Array{<:Number}, r:: Array{<:Number}; γ=γ) = (norm(r)^2+γ*norm(x)^2)/2

r(x:: Array{<:Number}; A=A, b=b) = A*x.-b

function step!(r:: Array{<:Number}, x:: Array{<:Number}, i:: Int64; A=A, Lₘₐₓ=Lₘₐₓ) 
		∇fᵢ = γ*x[i]
		@inbounds for j = A.colptr[i]:A.colptr[i+1]-1
			∇fᵢ += r[A.rowval[j]]*A.nzval[j]
		end

		# Atualização do iterado
		δ = -∇fᵢ/Lₘₐₓ
		x[i] += δ

		# Atualização do resíduo
		@inbounds for j = A.colptr[i]:A.colptr[i+1]-1
			r[A.rowval[j]] += δ*A.nzval[j]
		end
	end

function CD(x⁰:: Array{<:Number}, r:: Function, step!:: Function, f:: Function, ftarget:: Number, kₘₐₓ:: Int64)
	T = time()
	xᵏ = copy(x⁰)
	rᵏ = r(xᵏ) # Resíduo de x⁰

	# Histórico de iterados em valor objetivo
	fhist = Vector{Float64}(undef, kₘₐₓ+1) 
	fhist[1] = f(xᵏ, rᵏ)

	# Dimensão de x
	n = length(xᵏ)
	
	k = 1
	while true	
		for i = 1:n
			# Atualização do resíduo e iterado
			step!(rᵏ, xᵏ, rand(1:n))
		end

		# Update do histórico (só em iterações múltiplas de n)
		fxᵏ = f(xᵏ, rᵏ)
		fhist[k+1] = fxᵏ

		# Critério de parada
		if fxᵏ ≤ ftarget || k == kₘₐₓ
			return xᵏ, fhist[1:k+1], time()-T
		end
				
		k += 1
	end
end

x⁰ = zeros(size(A, 2));

kₘₐₓ = 10000

@time xCD, CDhist, tCD = CD(x⁰, r, step!, f, ftarget, kₘₐₓ)
