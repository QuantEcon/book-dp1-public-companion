using QuantEcon, LinearAlgebra, IterTools

function create_savings_model(; R=1.01, β=0.98, γ=2.5,  
                                w_min=0.01, w_max=10.0, w_size=100,
                                ρ=0.9, ν=0.1, z_size=20,
                                ϵ_min=-0.25, ϵ_max=0.25, ϵ_size=30)
    ϵ_grid = LinRange(ϵ_min, ϵ_max, ϵ_size)  
    ϕ = ones(ϵ_size) * (1 / ϵ_size)  # Uniform distribution
    w_grid = LinRange(w_min, w_max, w_size)  
    mc = tauchen(z_size, ρ, ν)
    z_grid, Q = exp.(mc.state_values), mc.p
    return (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q)
end


## == Functions for regular OPI == ##

"""
The function 

B(w, z, ϵ, w′) = 
    u(w + z + ϵ - w′/R) + β Σ v(w′, z′, ϵ′) Q(z, z′) ϕ(ϵ′)

"""
function B(i, j, k, l, v, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w, z, ϵ, w′ = w_grid[i], z_grid[j], ϵ_grid[k], w_grid[l]
    c = w + z + ϵ - (w′/ R)
    exp_value = 0.0
    for m in eachindex(z_grid)
        for n in eachindex(ϵ_grid)
            exp_value += v[l, m, n] * Q[j, m] * ϕ[n]
        end
    end
    return c > 0 ? c^(1-γ)/(1-γ) + β * exp_value : -Inf
end

"The policy operator."
function T_σ(v, σ, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w_idx, z_idx, ϵ_idx = (eachindex(g) for g in (w_grid, z_grid, ϵ_grid))
    v_new = similar(v)
    for (i, j, k) in product(w_idx, z_idx, ϵ_idx)
        v_new[i, j, k] = B(i, j, k, σ[i, j, k], v, model) 
    end
    return v_new
end

"Compute a v-greedy policy."
function get_greedy(v, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w_idx, z_idx, ϵ_idx = (eachindex(g) for g in (w_grid, z_grid, ϵ_grid))
    σ = Array{Int32}(undef, length(w_idx), length(z_idx), length(ϵ_idx))
    for (i, j, k) in product(w_idx, z_idx, ϵ_idx)
        _, σ[i, j, k] = findmax(B(i, j, k, l, v, model) for l in w_idx)
    end
    return σ
end


"Optimistic policy iteration routine."
function optimistic_policy_iteration(model; tolerance=1e-5, m=100)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    v = zeros(length(w_grid), length(z_grid), length(ϵ_grid))
    error = tolerance + 1
    while error > tolerance
        last_v = v
        σ = get_greedy(v, model)
        for i in 1:m
            v = T_σ(v, σ, model)
        end
        error = maximum(abs.(v - last_v))
        println("OPI current error = $error")
    end
    return get_greedy(v, model)
end




## == Functions for modified OPI == ##

"D(w, z, ϵ, w′, g) = u(w + z + ϵ - w′/R) + β g(z, w′)."
@inline function D(i, j, k, l, g, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w, z, ϵ, w′ = w_grid[i], z_grid[j], ϵ_grid[k], w_grid[l]
    c = w + z + ϵ - (w′ / R)
    return c > 0 ? c^(1-γ)/(1-γ) + β * g[j, l] : -Inf
end


"Compute a g-greedy policy."
function get_g_greedy(g, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w_idx, z_idx, ϵ_idx = (eachindex(g) for g in (w_grid, z_grid, ϵ_grid))
    σ = Array{Int32}(undef, length(w_idx), length(z_idx), length(ϵ_idx))
    for (i, j, k) in product(w_idx, z_idx, ϵ_idx)
        _, σ[i, j, k] = findmax(D(i, j, k, l, g, model) for l in w_idx)
    end
    return σ
end


"The modified policy operator."
function R_σ(g, σ, model)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    w_idx, z_idx, ϵ_idx = (eachindex(g) for g in (w_grid, z_grid, ϵ_grid))
    g_new = similar(g)
    for (j, i′) in product(z_idx, w_idx)  # j -> z, i′ -> w′ 
        out = 0.0
        for j′ in z_idx # j′ -> z′
            for k′ in ϵ_idx # k′ -> ϵ′
                # D(w′, z′, ϵ′, σ(w′, z′, ϵ′), g)
                out += D(i′, j′, k′, σ[i′, j′, k′], g, model) * 
                        Q[j, j′] * ϕ[k′]
            end
        end
        g_new[j, i′] = out
    end
    return g_new
end


"Modified optimistic policy iteration routine."
function mod_opi(model; tolerance=1e-5, m=100)
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    g = zeros(length(z_grid), length(w_grid))
    error = tolerance + 1
    while error > tolerance
        last_g = g
        σ = get_g_greedy(g, model)
        for i in 1:m
            g = R_σ(g, σ, model)
        end
        error = maximum(abs.(g - last_g))
        println("OPI current error = $error")
    end
    return get_g_greedy(g, model)
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16


function plot_contours(; savefig=false, 
                         figname="../figures/modified_opt_savings_1.pdf")

    model = create_savings_model()
    (; β, R, γ, ϵ_grid, ϕ, w_grid, z_grid, Q) = model
    σ_star = mod_opi(model)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    z_idx, ϵ_idx = eachindex(z_grid), eachindex(ϵ_grid)
    H = zeros(length(z_grid), length(ϵ_grid))

    w_indices = (1, length(w_grid))
    titles = "low wealth", "high wealth"
    for (ax, w_idx, title) in zip(axes, w_indices, titles)

        for (i_z, i_ϵ) in product(z_idx, ϵ_idx)
            w, z, ϵ = w_grid[w_idx], z_grid[i_z], ϵ_grid[i_ϵ]
            H[i_z, i_ϵ] = w_grid[σ_star[w_idx, i_z, i_ϵ]]
        end

        cs1 = ax.contourf(z_grid, ϵ_grid, transpose(H), alpha=0.5)
        #ctr1 = ax.contour(w_vals, z_vals, transpose(H), levels=[0.0])
        #plt.clabel(ctr1, inline=1, fontsize=13)
        plt.colorbar(cs1, ax=ax) #, format="%.6f")

        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(L"z", fontsize=fontsize)
        ax.set_ylabel(L"\varepsilon", fontsize=fontsize)
    end

    plt.tight_layout()
    if savefig
        fig.savefig(figname)
    end
    #plt.show()
end


