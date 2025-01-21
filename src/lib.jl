"""
    emd(μ, ν, C; kwargs...)

Compute the optimal transport plan for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost matrix `C` of size `(length(μ), length(ν))`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle.
```

This function is a wrapper of the function
[`emd`](https://pythonot.github.io/all.html#ot.emd) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> emd(μ, ν, C)
3×2 Matrix{Float64}:
 0.0  0.5
 0.0  0.2
 0.0  0.3
```

See also: [`emd2`](@ref)
"""
function emd(μ, ν, C; kwargs...)
    return pot.lp.emd(μ, ν, PyCall.PyReverseDims(permutedims(C)); kwargs...)
end

"""
    emd2(μ, ν, C; kwargs...)

Compute the optimal transport cost for the Monge-Kantorovich problem with source and target
marginals `μ` and `ν` and cost matrix `C` of size `(length(μ), length(ν))`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle.
```

This function is a wrapper of the function
[`emd2`](https://pythonot.github.io/all.html#ot.emd2) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> emd2(μ, ν, C)
0.95
```

See also: [`emd`](@ref)
"""
function emd2(μ, ν, C; kwargs...)
    return pot.lp.emd2(μ, ν, PyCall.PyReverseDims(permutedims(C)); kwargs...)
end

"""
    emd_1d(xsource, xtarget; kwargs...)

Compute the optimal transport plan for the Monge-Kantorovich problem with univariate
discrete measures with support `xsource` and `xtarget` as source and target marginals.

This function is a wrapper of the function
[`emd_1d`](https://pythonot.github.io/all.html#ot.emd_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [0.2, 0.5];

julia> xtarget = [0.8, 0.3];

julia> emd_1d(xsource, xtarget)
2×2 Matrix{Float64}:
 0.0  0.5
 0.5  0.0

julia> histogram_source = [0.8, 0.2];

julia> histogram_target = [0.7, 0.3];

julia> emd_1d(xsource, xtarget; a=histogram_source, b=histogram_target)
2×2 Matrix{Float64}:
 0.5  0.3
 0.2  0.0
```

See also: [`emd`](@ref), [`emd2_1d`](@ref)
"""
function emd_1d(xsource, xtarget; kwargs...)
    return pot.lp.emd_1d(xsource, xtarget; kwargs...)
end

"""
    emd2_1d(xsource, xtarget; kwargs...)

Compute the optimal transport cost for the Monge-Kantorovich problem with univariate
discrete measures with support `xsource` and `xtarget` as source and target marginals.

This function is a wrapper of the function
[`emd2_1d`](https://pythonot.github.io/all.html#ot.emd2_1d) in the Python Optimal Transport
package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [0.2, 0.5];

julia> xtarget = [0.8, 0.3];

julia> round(emd2_1d(xsource, xtarget); sigdigits=6)
0.05

julia> histogram_source = [0.8, 0.2];

julia> histogram_target = [0.7, 0.3];

julia> round(emd2_1d(xsource, xtarget; a=histogram_source, b=histogram_target); sigdigits=6)
0.201
```

See also: [`emd2`](@ref), [`emd2_1d`](@ref)
"""
function emd2_1d(xsource, xtarget; kwargs...)
    return pot.lp.emd2_1d(xsource, xtarget; kwargs...)
end

"""
    sinkhorn(μ, ν, C, ε; kwargs...)

Compute the optimal transport plan for the entropic regularization optimal transport problem
with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

This function is a wrapper of the function
[`sinkhorn`](https://pythonot.github.io/all.html#ot.sinkhorn) in the Python Optimal
Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest sinkhorn
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> sinkhorn(μ, ν, C, 0.01)
3×2 Matrix{Float64}:
 0.0  0.5
 0.0  0.2
 0.0  0.3
```

It is possible to provide multiple target marginals as columns of a matrix. In this case the
optimal transport costs are returned:

```jldoctest sinkhorn
julia> ν = [0.0 0.5; 1.0 0.5];

julia> round.(sinkhorn(μ, ν, C, 0.01); sigdigits=6)
2-element Vector{Float64}:
 0.95
 0.45
```

See also: [`sinkhorn2`](@ref)
"""
function sinkhorn(μ, ν, C, ε; kwargs...)
    return pot.sinkhorn(μ, ν, PyCall.PyReverseDims(permutedims(C)), ε; kwargs...)
end

"""
    sinkhorn2(μ, ν, C, ε; kwargs...)

Compute the optimal transport cost for the entropic regularization optimal transport problem
with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, and entropic regularization parameter `ε`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term.

This function is a wrapper of the function
[`sinkhorn2`](https://pythonot.github.io/all.html#ot.sinkhorn2) in the Python Optimal
Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest sinkhorn2
julia> μ = [0.5, 0.2, 0.3];

julia> ν = [0.0, 1.0];

julia> C = [0.0 1.0; 2.0 0.0; 0.5 1.5];

julia> round(sinkhorn2(μ, ν, C, 0.01); sigdigits=6)
0.95
```

It is possible to provide multiple target marginals as columns of a matrix.

```jldoctest sinkhorn2
julia> ν = [0.0 0.5; 1.0 0.5];

julia> round.(sinkhorn2(μ, ν, C, 0.01); sigdigits=6)
2-element Vector{Float64}:
 0.95
 0.45
```

See also: [`sinkhorn`](@ref)
"""
function sinkhorn2(μ, ν, C, ε; kwargs...)
    return pot.sinkhorn2(μ, ν, PyCall.PyReverseDims(permutedims(C)), ε; kwargs...)
end

"""
    empirical_sinkhorn_divergence(xsource, xtarget, ε; kwargs...)

Compute the Sinkhorn divergence from empirical data, where `xsource` and `xtarget` are
arrays representing samples in the source domain and target domain, respectively, and  `ε`
is the regularization term.

This function is a wrapper of the function
[`ot.bregman.empirical_sinkhorn_divergence`](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.empirical_sinkhorn_divergence)
in the Python Optimal Transport package. Keyword arguments are listed in the documentation of the Python function.

# Examples

```jldoctest
julia> xsource = [1.0];

julia> xtarget = [2.0, 3.0];

julia> ε = 0.01;

julia> empirical_sinkhorn_divergence(xsource, xtarget, ε) ≈
       sinkhorn2([1], [0.5, 0.5], [1.0 4.0], ε) -
       (
           sinkhorn2([1], [1], zeros(1, 1), ε) +
           sinkhorn2([0.5, 0.5], [0.5, 0.5], Float64[0 1; 1 0], ε)
       ) / 2
true
```

See also: [`sinkhorn2`](@ref)
"""
function empirical_sinkhorn_divergence(xsource, xtarget, ε; kwargs...)
    return pot.bregman.empirical_sinkhorn_divergence(
        reshape(xsource, Val(2)), reshape(xtarget, Val(2)), ε; kwargs...
    )
end

"""
    sinkhorn_unbalanced(μ, ν, C, ε, λ; kwargs...)

Compute the optimal transport plan for the unbalanced entropic regularization optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
term `λ`.

The optimal transport plan `γ` is of the same size as `C` and solves
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda \\mathrm{KL}(\\gamma 1, \\mu)
+ \\lambda \\mathrm{KL}(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\mathrm{KL}`` is the Kullback-Leibler divergence.

This function is a wrapper of the function
[`sinkhorn_unbalanced`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest sinkhorn_unbalanced
julia> μ = [0.5, 0.5];

julia> ν = [0.5, 0.5];

julia> C = [0.0 1.0; 1.0 0.0];

julia> round.(sinkhorn_unbalanced(μ, ν, C, 1, 1); sigdigits=7)
2×2 Matrix{Float64}:
 0.322054  0.118477
 0.118477  0.322054
```

```

See also: [`sinkhorn_unbalanced2`](@ref)
"""
function sinkhorn_unbalanced(μ, ν, C, ε, λ; kwargs...)
    return pot.sinkhorn_unbalanced(
        μ, ν, PyCall.PyReverseDims(permutedims(C)), ε, λ; kwargs...
    )
end

"""
    sinkhorn_unbalanced2(μ, ν, C, ε, λ; kwargs...)

Compute the optimal transport cost for the unbalanced entropic regularization optimal
transport problem with source and target marginals `μ` and `ν`, cost matrix `C` of size
`(length(μ), length(ν))`, entropic regularization parameter `ε`, and marginal relaxation
term `λ`.

The optimal transport cost is the scalar value
```math
\\inf_{\\gamma} \\langle \\gamma, C \\rangle
+ \\varepsilon \\Omega(\\gamma)
+ \\lambda \\mathrm{KL}(\\gamma 1, \\mu)
+ \\lambda \\mathrm{KL}(\\gamma^{\\mathsf{T}} 1, \\nu),
```
where ``\\Omega(\\gamma) = \\sum_{i,j} \\gamma_{i,j} \\log \\gamma_{i,j}`` is the entropic
regularization term and ``\\mathrm{KL}`` is the Kullback-Leibler divergence.

This function is a wrapper of the function
[`sinkhorn_unbalanced2`](https://pythonot.github.io/all.html#ot.sinkhorn_unbalanced2) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest sinkhorn_unbalanced2
julia> μ = [0.5, 0.1];

julia> ν = [0.5, 0.5];

julia> C = [0.0 1.0; 1.0 0.0];

julia> round.(sinkhorn_unbalanced2(μ, ν, C, 1., 1.); sigdigits=8)
0.19600125
```

See also: [`sinkhorn_unbalanced`](@ref)
"""
function sinkhorn_unbalanced2(μ, ν, C, ε, λ; kwargs...)
    return pot.sinkhorn_unbalanced2(
        μ, ν, PyCall.PyReverseDims(permutedims(C)), ε, λ; kwargs...
    )
end

"""
    barycenter(A, C, ε; kwargs...)

Compute the entropically regularized Wasserstein barycenter with histograms `A`, cost matrix
`C`, and entropic regularization parameter `ε`.

The Wasserstein barycenter is a histogram and solves
```math
\\inf_{a} \\sum_{i} W_{\\varepsilon,C}(a, a_i),
```
where the histograms ``a_i`` are columns of matrix `A` and ``W_{\\varepsilon,C}(a, a_i)``
is the optimal transport cost for the entropically regularized optimal transport problem
with marginals ``a`` and ``a_i``, cost matrix ``C``, and entropic regularization parameter
``\\varepsilon``. Optionally, weights of the histograms ``a_i`` can be provided with the
keyword argument `weights`.

This function is a wrapper of the function
[`barycenter`](https://pythonot.github.io/all.html#ot.barycenter) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> A = rand(10, 3);

julia> A ./= sum(A; dims=1);

julia> C = rand(10, 10);

julia> isapprox(sum(barycenter(A, C, 0.01; method="sinkhorn_stabilized")), 1; atol=1e-4)
true
```
"""
barycenter(A, C, ε; kwargs...) = pot.barycenter(A, C, ε; kwargs...)

"""
    barycenter_unbalanced(A, C, ε, λ; kwargs...)

Compute the entropically regularized unbalanced Wasserstein barycenter with histograms `A`, cost matrix
`C`, entropic regularization parameter `ε` and marginal relaxation parameter `λ`.

The Wasserstein barycenter is a histogram and solves
```math
\\inf_{a} \\sum_{i} W_{\\varepsilon,C,\\lambda}(a, a_i),
```
where the histograms ``a_i`` are columns of matrix `A` and ``W_{\\varepsilon,C,\\lambda}(a, a_i)}``
is the optimal transport cost for the entropically regularized optimal transport problem
with marginals ``a`` and ``a_i``, cost matrix ``C``, entropic regularization parameter
``\\varepsilon`` and marginal relaxation parameter ``\\lambda``. Optionally, weights of the histograms ``a_i`` can be provided with the
keyword argument `weights`.

This function is a wrapper of the function
[`barycenter_unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.barycenter_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> A = rand(10, 3);

julia> A ./= sum(A; dims=1);

julia> C = rand(10, 10);

julia> isapprox(sum(barycenter_unbalanced(A, C, 0.01, 1; method="sinkhorn_stabilized")), 1; atol=1e-4)
false

julia> isapprox(sum(barycenter_unbalanced(
           A, C, 0.01, 10_000; method="sinkhorn_stabilized", numItermax=5_000
       )), 1; atol=1e-4)
true
```

See also: [`barycenter`](@ref)
"""
function barycenter_unbalanced(A, C, ε, λ; kwargs...)
    return pot.barycenter_unbalanced(A, C, ε, λ; kwargs...)
end

"""
    gromov_wasserstein(μ, ν, Cμ, Cν, loss = "square_loss"; kwargs...)

Compute the exact Gromov-Wasserstein transport plan between `(μ, Cμ)` and `(ν, Cν)`.

The Gromov-Wasserstein transport problem seeks to find a minimizer of 
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, k, l} L((C_μ)_{ik}, (C_ν)_{jl}) \\gamma_{ij} \\gamma_{kl},
```
where ``L`` is quadratic (`loss = "square_loss"`) or the Kullback-Leibler divergence (`loss = "kl_loss"`).

This function is a wrapper of the function
[`gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.
"""
function gromov_wasserstein(μ, ν, Cμ, Cν, loss="square_loss"; kwargs...)
    return pot.gromov.gromov_wasserstein(Cμ, Cν, μ, ν, loss; kwargs...)
end

"""
    entropic_gromov_wasserstein(μ, ν, Cμ, Cν, ε, loss = "square_loss"; kwargs...)

Compute the entropy-regularized Gromov-Wasserstein transport plan between `(μ, Cμ)` and `(ν, Cν)` with parameter `ε`.

The entropy-regularized Gromov-Wasserstein transport problem seeks to find a minimizer of 
```math
\\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, k, l} L((C_μ)_{ik}, (C_ν)_{jl}) \\gamma_{ij} \\gamma_{kl} + ε \\Omega(\\gamma),
```
where ``L`` is quadratic (`loss = "square_loss"`) or the Kullback-Leibler divergence (`loss = "kl_loss"`)
and ``\\Omega(\\gamma) = \\sum_{ij} \\gamma_{ij} \\log(\\gamma_{ij})`` is the entropic regularization term. 

This function is a wrapper of the function
[`entropic_gromov_wasserstein`](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_gromov_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.
"""
function entropic_gromov_wasserstein(μ, ν, Cμ, Cν, ε, loss="square_loss"; kwargs...)
    return pot.gromov.entropic_gromov_wasserstein(Cμ, Cν, μ, ν, loss, ε; kwargs...)
end

"""
    mm_unbalanced(a, b, M, reg_m; reg=0, c=a*b', kwargs...)

Solve the unbalanced optimal transport problem and return the OT plan.
The function solves the following optimization problem:

```math
W = \\min_{\\gamma \\geq 0} \\langle \\gamma, M \\rangle_F +
    \\mathrm{reg_{m1}} \\cdot \\operatorname{div}(\\gamma \\mathbf{1}, a) +
    \\mathrm{reg_{m2}} \\cdot \\operatorname{div}(\\gamma^\\mathsf{T} \\mathbf{1}, b) +
    \\mathrm{reg} \\cdot \\operatorname{div}(\\gamma, c)
```

where

- `M` is the metric cost matrix,
- `a` and `b` are source and target unbalanced distributions,
- `c` is a reference distribution for the regularization,
- `reg_m` is the marginal relaxation term (if it is a scalar or an indexable object of length 1, then the same term is applied to both marginal relaxations), and
- `reg` is a regularization term.

This function is a wrapper of the function
[`mm_unbalanced`](https://pythonot.github.io/gen_modules/ot.unbalanced.html#ot.unbalanced.mm_unbalanced) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.

# Examples

```jldoctest
julia> a=[.5, .5];

julia> b=[.5, .5];

julia> M=[1. 36.; 9. 4.];

julia> round.(mm_unbalanced(a, b, M, 5, div="kl"), digits=2)
2×2 Matrix{Float64}:
 0.45  0.0
 0.0   0.34

julia> round.(mm_unbalanced(a, b, M, 5, div="l2"), digits=2)
2×2 Matrix{Float64}:
 0.4  0.0
 0.0  0.1
```

"""
function mm_unbalanced(a, b, M, reg_m; kwargs...)
    return pot.unbalanced.mm_unbalanced(a, b, M, reg_m; kwargs...)
end


"""
    entropic_partial_wasserstein(a, b, M, reg; kwargs...)

Solves the partial optimal transport problem and returns the OT plan
The function considers the following problem:

```math
\\gamma = \\mathop{\\arg \\min}_\\gamma \\quad \\langle \\gamma,
\\mathbf{M} \\rangle_F + \\mathrm{reg} \\cdot\\Omega(\\gamma)
```

- `a` and `b` are the sample weights
- `M` is the metric cost matrix
- `reg` is a regularization term > 0

This function is a wrapper of the function
[`entropic_partial_wasserstein`](https://pythonot.github.io/gen_modules/ot.partial.html#ot.partial.entropic_partial_wasserstein) in the
Python Optimal Transport package. Keyword arguments are listed in the documentation of the
Python function.


# Examples

```jldoctest
julia> a = [.1, .2];

julia> b = [.1, .1];

julia> M = [0. 1.; 2. 3.];

julia> round.(entropic_partial_wasserstein(a, b, M, 1, m=0.1), digits=2)
2×2 Matrix{Float64}:
 0.06  0.02
 0.01  0.0
```

"""
function entropic_partial_wasserstein(a, b, M, reg; kwargs...)
    return pot.partial.entropic_partial_wasserstein(a, b, M, reg; kwargs...)
end
