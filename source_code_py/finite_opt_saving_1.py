import numpy as np
from finite_opt_saving_0 import U, B
from numba import njit, prange
from itertools import product

@njit
def cartesian_product(arrays):
    arrays = [np.asarray(a) for a in arrays]
    num_arrays = len(arrays)
    lengths = np.zeros(num_arrays, dtype=np.int64)
    for i in prange(num_arrays):
        lengths[i] = len(arrays[i])
    num_elements = np.prod(lengths)

    result = np.zeros((num_elements, num_arrays), dtype=arrays[0].dtype)
    temp_product = np.zeros(num_arrays, dtype=np.int64)

    for i in prange(num_elements):
        for j in prange(num_arrays):
            result[i, j] = arrays[j][temp_product[j]]
        
        # Increment the temporary product
        for j in prange(num_arrays - 1, -1, -1):
            temp_product[j] += 1
            if temp_product[j] == lengths[j] and j != 0:
                temp_product[j] = 0
            else:
                break
    return result


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
    w_idx, y_idx = np.arange(len(w_grid)), np.arange(len(y_grid))
    wn, yn = len(w_grid), len(y_grid)
    n = wn * yn
    # Build P_σ and r_σ as multi-index arrays
    P_σ = np.zeros((wn, yn, wn, yn))
    r_σ = np.zeros((wn, yn))
    prd = cartesian_product([w_idx, y_idx])
    for (i, j) in prd:
        w, y, w_1 = w_grid[i], y_grid[j], w_grid[σ[i, j]]
        r_σ[i, j] = U(w + y - w_1/R, γ)
        for (i_1, j_1) in prd:
            if i_1 == σ[i, j]:
                P_σ[i, j, i_1, j_1] = Q[j, j_1]
    # Solve for the value of σ
    P_σ = P_σ.reshape(n, n)
    r_σ = r_σ.reshape(n)

    I = np.identity(n)
    v_σ = np.linalg.solve((I - β * P_σ), r_σ)
    # Return as multi-index array
    v_σ = v_σ.reshape(wn, yn)
    return v_σ

