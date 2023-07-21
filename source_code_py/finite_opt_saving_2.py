from quantecon import compute_fixed_point

import numpy as np
from numba import njit
import time
from finite_opt_saving_1 import get_greedy, get_value
from finite_opt_saving_0 import create_savings_model, T, T_σ



def value_iteration(model, tol=1e-5):
    """Value function iteration routine."""
    vz = np.zeros((len(model.w_grid), len(model.y_grid)))
    v_star = compute_fixed_point(lambda v: T(v, model), vz,
                                 error_tol=tol, max_iter=1000, print_skip=25)
    return get_greedy(v_star, model)


@njit
def policy_iteration(model):
    """Howard policy iteration routine."""
    wn, yn = len(model.w_grid), len(model.y_grid)
    σ = np.ones((wn, yn), dtype=np.int32)
    i, error = 0, 1.0
    while error > 0:
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = np.max(np.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error: {error}.")
    return σ


@njit
def optimistic_policy_iteration(model, tolerance=1e-5, m=100):
    """Optimistic policy iteration routine."""
    v = np.zeros((len(model.w_grid), len(model.y_grid)))
    error = tolerance + 1
    while error > tolerance:
        last_v = v
        σ = get_greedy(v, model)
        for i in range(0, m):
            v = T_σ(v, σ, model)
        error = np.max(np.abs(v - last_v))
    return get_greedy(v, model)


# Plots

import matplotlib.pyplot as plt


def plot_timing(m_vals=np.arange(1, 601, 10),
                savefig=False):
    model = create_savings_model(y_size=5)
    print("Running Howard policy iteration.")
    t1 = time.time()
    σ_pi = policy_iteration(model)
    pi_time = time.time() - t1
    print(f"PI completed in {pi_time} seconds.")
    print("Running value function iteration.")
    t1 = time.time()
    σ_vfi = value_iteration(model)
    vfi_time = time.time() - t1
    print(f"VFI completed in {vfi_time} seconds.")

    assert np.allclose(σ_vfi, σ_pi), "Warning: policies deviated."

    opi_times = []
    for m in m_vals:
        print(f"Running optimistic policy iteration with m={m}.")
        t1 = time.time()
        σ_opi = optimistic_policy_iteration(model, m=m)
        t2 = time.time()
        assert np.allclose(σ_opi, σ_pi), "Warning: policies deviated."
        print(f"OPI with m={m} completed in {t2-t1} seconds.")
        opi_times.append(t2-t1)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m_vals, [vfi_time]*len(m_vals),
            linewidth=2, label="value function iteration")
    ax.plot(m_vals, [pi_time]*len(m_vals),
            linewidth=2, label="Howard policy iteration")
    ax.plot(m_vals, opi_times, linewidth=2,
            label="optimistic policy iteration")
    ax.legend(frameon=False)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("time")
    #plt.show()
    if savefig:
        fig.savefig("../figures/finite_opt_saving_2_1.png")
    return (pi_time, vfi_time, opi_times)


def plot_policy(method="pi", savefig=False):
    model = create_savings_model()
    β, R, γ, w_grid, y_grid, Q = model
    if method == "vfi":
        σ_star =  value_iteration(model)
    elif method == "pi":
        σ_star = policy_iteration(model)
    else:
        method = "OPT"
        σ_star = optimistic_policy_iteration(model)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(w_grid, w_grid, "k--", label=r"$45$")
    ax.plot(w_grid, w_grid[σ_star[:, 0]], label=r"$\sigma^*(\cdot, y_1)$")
    ax.plot(w_grid, w_grid[σ_star[:, -1]], label=r"$\sigma^*(\cdot, y_N)$")
    ax.legend()
    plt.title(f"Method: {method}")
    #plt.show()
    if savefig:
        fig.savefig(f"../figures/finite_opt_saving_2_2_{method}.png")
