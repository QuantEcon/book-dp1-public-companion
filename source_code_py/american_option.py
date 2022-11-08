"""
Valuation for finite-horizon American call options in discrete time.

"""

from quantecon.markov import tauchen, MarkovChain
import numpy as np
from collections import namedtuple
from s_approx import successive_approx
from itertools import product


# NamedTuple Model
Model = namedtuple("Model", ("t_vals", "z_vals","w_vals", "Q",
                             "φ", "T", "β", "K", "e"))


def create_american_option_model(
        n=100, μ=10.0,   # Markov state grid size and mean value
        ρ=0.98, ν=0.2,   # persistence and volatility for Markov state
        s=0.3,           # volatility parameter for W_t
        r=0.01,          # interest rate
        K=10.0, T=200):  # strike price and expiration date
    """
    Creates an instance of the option model with log S_t = Z_t + W_t.
    """
    t_vals = np.arange(T+1)
    mc = tauchen(ρ, ν, n=n)
    z_vals, Q = mc.state_values + μ, mc.P
    w_vals, φ, β = [-s, s], [0.5, 0.5], 1 / (1 + r)
    e = lambda t, i_w, i_z: (t <= T) * (z_vals[i_z] + w_vals[i_w] - K)
    return Model(t_vals=t_vals, z_vals=z_vals, w_vals=w_vals, Q=Q,
                 φ=φ, T=T, β=β, K=K, e=e)


def C(h, model):
    """The continuation value operator."""
    t_vals, z_vals, w_vals, Q, φ, T, β, K, e = model
    Ch = np.empty_like(h)
    z_idx, w_idx = range(len(z_vals)), range(len(w_vals))
    for (t, i_z) in product(t_vals, z_idx):
        out = 0.0
        for (i_w_1, i_z_1) in product(w_idx, z_idx):
            t_1 = min(t + 1, T)
            out += max(e(t_1, i_w_1, i_z_1), h[t_1, i_z_1]) * \
                        Q[i_z, i_z_1] * φ[i_w_1]
        Ch[t, i_z] = β * out
    return Ch



def compute_cvf(model):
    """
    Compute the continuation value function by successive approx.
    """
    h_init = np.zeros((len(model.t_vals), len(model.z_vals)))
    h_star = successive_approx(lambda h: C(h, model), h_init)
    return h_star


# Plots


import matplotlib.pyplot as plt


def plot_contours(savefig=False,
                  figname="../figures/american_option_1.png"):

    model = create_american_option_model()
    t_vals, z_vals, w_vals, Q, φ, T, β, K, e = model
    h_star = compute_cvf(model)
    fig, axes = plt.subplots(3, 1, figsize=(7, 11))
    z_idx, w_idx = range(len(z_vals)), range(len(w_vals))
    H = np.zeros((len(w_vals), len(z_vals)))
    for (ax_index, t) in zip(range(3), (1, 195, 199)):

        ax = axes[ax_index]

        for (i_w, i_z) in product(w_idx, z_idx):
            H[i_w, i_z] = e(t, i_w, i_z) - h_star[t, i_z]

        cs1 = ax.contourf(w_vals, z_vals, np.transpose(H), alpha=0.5)
        ctr1 = ax.contour(w_vals, z_vals, np.transpose(H), levels=[0.0])
        plt.clabel(ctr1, inline=1, fontsize=13)
        plt.colorbar(cs1, ax=ax)

        ax.set_title(f"$t={t}$")
        ax.set_xlabel(r"$w$")
        ax.set_ylabel(r"$z$")

    fig.tight_layout()
    if savefig:
        fig.savefig(figname)
    plt.show()


def plot_strike(savefig=False,
                fontsize=12,
                figname="../figures/american_option_2.png"):
    model = create_american_option_model()
    t_vals, z_vals, w_vals, Q, φ, T, β, K, e = model
    h_star = compute_cvf(model)

    # Built Markov chains for simulation
    z_mc = MarkovChain(Q, z_vals)
    P_φ = np.zeros((len(w_vals), len(w_vals)))
    for i in range(len(w_vals)):  # Build IID chain
        P_φ[i, :] = φ
    w_mc = MarkovChain(P_φ, w_vals)
    y_min = np.min(z_vals) + np.min(w_vals)
    y_max = np.max(z_vals) + np.max(w_vals)
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    for ax in axes:

        # Generate price series
        z_draws = z_mc.simulate_indices(T, init=int(len(z_vals) / 2 - 10))
        w_draws = w_mc.simulate_indices(T)
        s_vals = np.empty_like(z_draws)
        for idx in range(T):
            s_vals[idx] = z_vals[z_draws[idx]] + w_vals[w_draws[idx]]

        # Find the exercise date, if any.
        exercise_date = T + 1
        for t in range(T):
            if e(t, w_draws[t], z_draws[t]) >= h_star[w_draws[t], z_draws[t]]:
                exercise_date = t

        assert exercise_date < T, "Option not exercised."

        # Plot
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, T+1)
        ax.fill_between(range(T), np.ones(T) * K, np.ones(T) * y_max, alpha=0.2)
        ax.plot(range(T), s_vals, label=r"$S_t$")
        ax.plot((exercise_date,), (s_vals[exercise_date]), "ko")
        ax.vlines((exercise_date,), 0, (s_vals[exercise_date]), ls="--", colors="black")
        ax.legend(loc="upper left", fontsize=fontsize)
        ax.text(-10, 11, "in the money", fontsize=fontsize, rotation=90)
        ax.text(-10, 7.2, "out of the money", fontsize=fontsize, rotation=90)
        ax.text(exercise_date-20, 6, #s_vals[exercise_date]+0.8,
                "exercise date", fontsize=fontsize)
        ax.set_xticks((1, T))
        ax.set_yticks((y_min, y_max))

    if savefig:
        fig.savefig(figname)
    plt.show()
