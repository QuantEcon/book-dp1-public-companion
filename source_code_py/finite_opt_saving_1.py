import numpy as np
from finite_opt_saving_0 import U, B
from numba import njit, prange


@njit(parallel=True)
def get_greedy(v, model):
    """Compute a v-greedy policy."""
    β, R, γ, w_grid, y_grid, Q = model
    σ = np.empty((w_grid.shape[0], y_grid.shape[0]), dtype=np.int32)
    for i in prange(w_grid.shape[0]):
        for j in prange(y_grid.shape[0]):
            x_tmp = np.array([B(i, j, k, v, model) for k in
                             np.arange(w_grid.shape[0])])
            σ[i, j] = np.argmax(x_tmp)
    return σ


@njit
def single_to_multi(m, yn):
    # Function to extract (i, j) from m = i + (j-1)*yn
    return (m//yn, m%yn)


@njit(parallel=True)
def get_value(σ, model):
    """Get the value v_σ of policy σ."""
    # Unpack and set up
    β, R, γ, w_grid, y_grid, Q = model
    wn, yn = len(w_grid), len(y_grid)
    n = wn * yn
    # Allocate and create single index versions of P_σ and r_σ
    P_σ = np.zeros((n, n))
    r_σ = np.zeros(n)
    for m in prange(n):
        i, j = single_to_multi(m, yn)
        w, y, w_1 = w_grid[i], y_grid[j], w_grid[σ[i, j]]
        r_σ[m] = U(w + y - w_1/R, γ)
        for m_1 in prange(n):
            i_1, j_1 = single_to_multi(m_1, yn)
            if i_1 == σ[i, j]:
                P_σ[m, m_1] = Q[j, j_1]

    # Solve for the value of σ
    I = np.identity(n)
    v_σ = np.linalg.solve((I - β * P_σ), r_σ)
    # Return as multi-index array
    v_σ = v_σ.reshape(wn, yn)
    return v_σ
