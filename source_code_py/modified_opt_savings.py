from quantecon import tauchen

import numpy as np
from collections import namedtuple
from numba import njit, prange


# NamedTuple Model
Model = namedtuple("Model", ("β", "R", "γ", "ϵ_grid", "φ",
                             "w_grid", "z_grid", "Q"))


def create_savings_model(R=1.01, β=0.98, γ=2.5,
                         w_min=0.01, w_max=10.0, w_size=100,
                         ρ=0.9, ν=0.1, z_size=20,
                         ϵ_min=-0.25, ϵ_max=0.25, ϵ_size=30):
    ϵ_grid = np.linspace(ϵ_min, ϵ_max, ϵ_size)
    φ = np.ones(ϵ_size) * (1 / ϵ_size)  # Uniform distributoin
    w_grid = np.linspace(w_min, w_max, w_size)
    mc = tauchen(ρ, ν, n=z_size)
    z_grid, Q = np.exp(mc.state_values), mc.P
    return Model(β=β, R=R, γ=γ, ϵ_grid=ϵ_grid, φ=φ, w_grid=w_grid,
                 z_grid=z_grid, Q=Q)


## == Functions for regular OPI == ##


@njit(parallel=True)
def B(i, j, k, l, v, model):
    """
    The function

    B(w, z, ϵ, w′) =
        u(w + z + ϵ - w′/R) + β Σ v(w′, z′, ϵ′) Q(z, z′) ϕ(ϵ′)

    """
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w, z, ϵ, w_1 = w_grid[i], z_grid[j], ϵ_grid[k], w_grid[l]
    c = w + z + ϵ - (w_1/ R)
    exp_value = 0.0
    for m in prange(len(z_grid)):
        for n in prange(len(ϵ_grid)):
            exp_value += v[l, m, n] * Q[j, m] * φ[n]
    return c**(1-γ)/(1-γ) + β * exp_value if c > 0 else -np.inf


@njit(parallel=True)
def T_σ(v, σ, model):
    """The policy operator."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    v_new = np.empty_like(v)
    for i in prange(len(w_grid)):
        for j in prange(len(z_grid)):
            for k in prange(len(ϵ_grid)):
                v_new[i, j, k] = B(i, j, k, σ[i, j, k], v, model)
    return v_new


@njit(parallel=True)
def get_greedy(v, model):
    """Compute a v-greedy policy."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w_n, z_n, ϵ_n = len(w_grid), len(z_grid), len(ϵ_grid)
    σ = np.empty((w_n, z_n, ϵ_n), dtype=np.int32)
    for i in prange(w_n):
        for j in prange(z_n):
            for k in prange(ϵ_n):
                _tmp = np.array([B(i, j, k, l, v, model) for l
                                in range(w_n)])
                σ[i, j, k] = np.argmax(_tmp)
    return σ


def optimistic_policy_iteration(model, tolerance=1e-5, m=100):
    """Optimistic policy iteration routine."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w_n, z_n, ϵ_n = len(w_grid), len(z_grid), len(ϵ_grid)
    v = np.zeros((w_n, z_n, ϵ_n))
    error = tolerance + 1
    while error > tolerance:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(m):
            v = T_σ(v, σ, model)
        error = np.max(np.abs(v - last_v))
        print(f"OPI current error = {error}")
    return get_greedy(v, model)




## == Functions for modified OPI == ##


@njit
def D(i, j, k, l, g, model):
    """D(w, z, ϵ, w′, g) = u(w + z + ϵ - w′/R) + β g(z, w′)."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w, z, ϵ, w_1 = w_grid[i], z_grid[j], ϵ_grid[k], w_grid[l]
    c = w + z + ϵ - (w_1 / R)
    return c**(1-γ)/(1-γ) + β * g[j, l] if c > 0 else -np.inf


@njit(parallel=True)
def get_g_greedy(g, model):
    """Compute a g-greedy policy."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w_n, z_n, ϵ_n = len(w_grid), len(z_grid), len(ϵ_grid)
    σ = np.empty((w_n, z_n, ϵ_n), dtype=np.int32)
    for i in prange(w_n):
        for j in prange(z_n):
            for k in prange(ϵ_n):
                _tmp = np.array([D(i, j, k, l, g, model) for l
                                in range(w_n)])
                σ[i, j, k] = np.argmax(_tmp)
    return σ


@njit(parallel=True)
def R_σ(g, σ, model):
    """The modified policy operator."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    w_n, z_n, ϵ_n = len(w_grid), len(z_grid), len(ϵ_grid)
    g_new = np.empty_like(g)
    for j in prange(z_n):
        for i_1 in prange(w_n):
            out = 0.0
            for j_1 in prange(z_n):
                for k_1 in prange(ϵ_n):
                    out += D(i_1, j_1, k_1, σ[i_1, j_1, k_1], g,
                             model) * Q[j, j_1] * φ[k_1]
            g_new[j, i_1] = out
    return g_new


def mod_opi(model, tolerance=1e-5, m=100):
    """Modified optimistic policy iteration routine."""
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    g = np.zeros((len(z_grid), len(w_grid)))
    error = tolerance + 1
    while error > tolerance:
        last_g = g
        σ = get_g_greedy(g, model)
        for i in range(m):
            g = R_σ(g, σ, model)
        error = np.max(np.abs(g - last_g))
        print(f"OPI current error = {error}")
    return get_g_greedy(g, model)


# Plots


import matplotlib.pyplot as plt


def plot_contours(savefig=False,
                  figname="../figures/modified_opt_savings_1.pdf"):

    model = create_savings_model()
    β, R, γ, ϵ_grid, φ, w_grid, z_grid, Q = model
    σ_star = mod_opi(model)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    z_n, ϵ_n = len(z_grid), len(ϵ_grid)
    z_idx, ϵ_idx = np.arange(z_n), np.arange(ϵ_n)
    H = np.zeros((z_n, ϵ_n))

    w_indices = (0, len(w_grid)-1)
    titles = "low wealth", "high wealth"
    for (ax, w_idx, title) in zip(axes, w_indices, titles):

        for i_z in z_idx:
            for i_ϵ in ϵ_idx:
                w, z, ϵ = w_grid[w_idx], z_grid[i_z], ϵ_grid[i_ϵ]
                H[i_z, i_ϵ] = w_grid[σ_star[w_idx, i_z, i_ϵ]]

        cs1 = ax.contourf(z_grid, ϵ_grid, np.transpose(H), alpha=0.5)
        #ctr1 = ax.contour(w_vals, z_vals, transpose(H), levels=[0.0])
        #plt.clabel(ctr1, inline=1, fontsize=13)
        plt.colorbar(cs1, ax=ax) #, format="%.6f")

        ax.set_title(title)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\varepsilon$")

    plt.tight_layout()
    if savefig:
        fig.savefig(figname)
    plt.show()
