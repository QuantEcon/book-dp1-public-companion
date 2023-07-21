"""

Inventory management model with state-dependent discounting.  The discount
factor takes the form β_t = Z_t, where (Z_t) is a discretization of a
Gaussian AR(1) process

    X_t = ρ X_{t-1} + b + ν W_t.

"""

from quantecon import compute_fixed_point
from quantecon.markov import tauchen, MarkovChain

import numpy as np
from numba import njit, prange
from collections import namedtuple

# NamedTuple Model
Model = namedtuple("Model", ("K", "c", "κ", "p", "z_vals", "Q"))


@njit
def demand_pdf(p, d):
    return (1 - p)**d * p


def create_sdd_inventory_model(
        ρ=0.98, ν=0.002, n_z=20, b=0.97,   # Z state parameters
        K=40, c=0.2, κ=0.8, p=0.6):        # firm and demand parameters
    mc = tauchen(ρ, ν, n=n_z)
    z_vals, Q = mc.state_values + b, mc.P
    rL = np.max(np.abs(np.linalg.eigvals(z_vals * Q)))
    assert rL < 1, "Error: r(L) >= 1."    # check r(L) < 1
    return Model(K=K, c=c, κ=κ, p=p, z_vals=z_vals, Q=Q)


@njit
def B(x, i_z, a, v, model, d_max=101):
    """
    The function B(x, z, a, v) = r(x, a) + β(z) Σ_x′ v(x′) P(x, a, x′).
    """
    K, c, κ, p, z_vals, Q = model
    z = z_vals[i_z]
    _tmp = np.array([np.minimum(x, d)*demand_pdf(p, d) for d in range(d_max)])
    reward = np.sum(_tmp) - c * a - κ * (a > 0)
    cv = 0.0
    for i_z_1 in prange(len(z_vals)):
        _tmp = np.array([v[np.maximum(x - d, 0) + a, i_z_1] * demand_pdf(p, d)
                            for d in range(d_max)])
        cv += np.sum(_tmp) * Q[i_z, i_z_1]
    return reward + z * cv


@njit(parallel=True)
def T(v, model):
    """The Bellman operator."""
    K, c, κ, p, z_vals, Q = model
    new_v = np.empty_like(v)
    for i_z in prange(len(z_vals)):
        for x in prange(K+1):
            _tmp = np.array([B(x, i_z, a, v, model)
                             for a in range(K-x+1)])
            new_v[x, i_z] = np.max(_tmp)
    return new_v


@njit(parallel=True)
def get_greedy(v, model):
    """Get a v-greedy policy.  Returns a zero-based array."""
    K, c, κ, p, z_vals, Q = model
    n_z = len(z_vals)
    σ_star = np.zeros((K+1, n_z), dtype=np.int32)
    for i_z in prange(n_z):
        for x in range(K+1):
            _tmp = np.array([B(x, i_z, a, v, model)
                             for a in range(K-x+1)])
            σ_star[x, i_z] = np.argmax(_tmp)
    return σ_star



def solve_inventory_model(v_init, model):
    """Use successive_approx to get v_star and then compute greedy."""
    v_star = compute_fixed_point(lambda v: T(v, model), v_init,
                                 error_tol=1e-5, max_iter=1000, print_skip=25)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star


# == Plots == #

import matplotlib.pyplot as plt


# Create an instance of the model and solve it
model = create_sdd_inventory_model()
K, c, κ, p, z_vals, Q = model
n_z = len(z_vals)
v_init = np.zeros((K+1, n_z), dtype=float)
print("Solving model.")
v_star, σ_star = solve_inventory_model(v_init, model)
z_mc = MarkovChain(Q, z_vals)


def sim_inventories(ts_length, X_init=0):
    """Simulate given the optimal policy."""
    global p, z_mc
    i_z = z_mc.simulate_indices(ts_length, init=1)
    X = np.zeros(ts_length, dtype=np.int32)
    X[0] = X_init
    rand = np.random.default_rng().geometric(p=p, size=ts_length-1) - 1
    for t in range(ts_length-1):
        X[t+1] = np.maximum(X[t] - rand[t], 0) + σ_star[X[t], i_z[t]]
    return X, z_vals[i_z]


def plot_ts(ts_length=400,
                 fontsize=10,
                 figname="../figures/inventory_sdd_ts.pdf",
                 savefig=False):
    X, Z = sim_inventories(ts_length)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))

    ax = axes[0]
    ax.plot(X, label=r"$X_t$", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel("inventory", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, np.max(X)+3)

    # calculate interest rate from discount factors
    r = (1 / Z) - 1

    ax = axes[1]
    ax.plot(r, label=r"$r_t$", alpha=0.7)
    ax.set_xlabel(r"$t$", fontsize=fontsize)
    ax.set_ylabel("interest rate", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)

    plt.tight_layout()
    #plt.show()
    if savefig:
        fig.savefig(figname)
