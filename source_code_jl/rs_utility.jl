include("s_approx.jl")
using LinearAlgebra, QuantEcon

function create_rs_utility_model(;
        n=180,      # size of state space
        β=0.95,     # time discount factor
        ρ=0.96,     # correlation coef in AR(1)
        σ=0.1,      # volatility
        θ=-1.0)     # risk aversion
    mc = tauchen(n, ρ, σ, 0, 10)  # n_std = 10
    x_vals, P = mc.state_values, mc.p 
    r = x_vals      # special case u(c(x)) = x
    return (; β, θ, ρ, σ, r, x_vals, P)
end

function K(v, model)
    (; β, θ, ρ, σ, r, x_vals, P) = model
    return r + (β/θ) * log.(P * (exp.(θ*v)))
end

function compute_rs_utility(model)
    (; β, θ, ρ, σ, r, x_vals, P) = model
    v_init = zeros(length(x_vals))
    v_star = successive_approx(v -> K(v, model), 
                               v_init, tolerance=1e-10)
    return v_star
end


# Plots

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering
fontsize=16

function plot_v(; savefig=false, 
                  figname="figures/rs_utility_1.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    model = create_rs_utility_model()
    (; β, θ, ρ, σ, r, x_vals, P) = model

    a = 1/(1 - (ρ*β))
    b = (β /(1 - β)) * (θ/2) * (a*σ)^2 
    
    v_star = compute_rs_utility(model)
    v_star_a = a * x_vals .+ b
    ax.plot(x_vals, v_star, lw=2, alpha=0.7, label="approximate fixed point")
    ax.plot(x_vals, v_star_a, "k--", lw=2, alpha=0.7, label=L"v(x)=ax + b")
    ax.set_xlabel(L"x", fontsize=fontsize)

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end




function plot_multiple_v(; savefig=false, 
                  figname="figures/rs_utility_2.pdf")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    σ_vals = 0.05, 0.1

    for σ in σ_vals
        model = create_rs_utility_model(σ=σ)
        (; β, θ, r, x_vals, P) = model
        v_star = compute_rs_utility(model)
        ax.plot(x_vals, v_star, lw=2, alpha=0.7, label=L"\sigma="*"$σ")
        ax.set_xlabel(L"x", fontsize=fontsize)
        ax.set_ylabel(L"v(x)", fontsize=fontsize)
    end

    ax.legend(frameon=false, fontsize=fontsize, loc="upper left")
    plt.show()
    if savefig
        fig.savefig(figname)
    end
end

