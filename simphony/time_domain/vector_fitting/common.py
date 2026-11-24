"""Domain-independent pieces of vector fitting, shared by `s_domain` and `z_domain`.

Each domain builds its own basis matrices: `phi1[k, n] = 1 / (x_k - p_n)`, with `x = s` (continuous) or `x = z`
(discrete), and `phi0 = [1, phi1]`. It also applies its own stability correction after pole relocation. Everything
else in the fitting loop is the same linear algebra in both domains and lives here.
"""

import gc
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _clear_vector_fitting_memory():
    """Release local Python references and JAX compilation/device caches."""
    gc.collect()
    jax.clear_caches()


def _lstsq_matrices(model_order, transfer_function, phi0, phi1):
    """Build the reduced least-squares system used for pole relocation.

    This is the fast vector fitting reduction: for each input/output pair, a
    Householder QR of ``[phi0, -diag(H) phi1]`` eliminates the pair's own
    numerator coefficients, leaving the ``R22`` block and ``Q2^H H`` for the
    shared weight coefficients. ``diag(H) phi1`` is applied as a row scaling,
    so no dense diagonal matrix is built.

    A full QR per pair is used instead of factoring ``phi0`` once and projecting
    each block against it: that single classical Gram-Schmidt pass loses
    orthogonality once a fit approaches machine precision, and the pole
    relocation then drifts (s-domain fits fell from -255 dB to -73 dB over 20
    iterations).
    """
    num_outputs = transfer_function.shape[1]
    num_inputs = transfer_function.shape[2]
    num_pairs = num_inputs * num_outputs
    numerator_size = phi0.shape[1]
    dtype = jnp.result_type(transfer_function, phi0, phi1)
    M = jnp.zeros((num_pairs * model_order, model_order), dtype=dtype)
    B = jnp.zeros((num_pairs * model_order,), dtype=dtype)

    transfer_pairs = (
        transfer_function.transpose(2, 1, 0)
        .reshape(
            num_pairs,
            transfer_function.shape[0],
        )
        .T
    )

    def process_pair(pair_index, carry):
        M, B = carry
        V = transfer_pairs[:, pair_index]
        Q, R = jnp.linalg.qr(jnp.hstack([phi0, -phi1 * V[:, None]]), mode="reduced")
        R22 = R[numerator_size:, numerator_size:]
        b_block = Q[:, numerator_size:].conj().T @ V
        row_start = pair_index * model_order

        M = jax.lax.dynamic_update_slice(M, R22, (row_start, 0))
        B = jax.lax.dynamic_update_slice(B, b_block, (row_start,))
        return M, B

    M, B = jax.lax.fori_loop(0, num_pairs, process_pair, (M, B))

    return M, B


def _weight_error(phi1, weight_coeffs):
    """RMS deviation of the weighting function `1 + phi1 @ weight_coeffs` from 1."""
    weights_minus_one = phi1 @ weight_coeffs
    return jnp.sqrt(jnp.mean(jnp.abs(weights_minus_one) ** 2))


def _relocate_poles(poles, weight_coeffs):
    """Zeros of the weighting function, i.e. the new poles before any stability correction."""
    model_order = poles.shape[0]
    relocated_poles, _ = jnp.linalg.eig(
        jnp.diag(poles) - jnp.outer(jnp.ones(model_order), weight_coeffs)
    )
    return relocated_poles


def _fit_to_poles(transfer_function, phi0):
    """Solve for residues and feedthrough of every input/output pair with the poles fixed.

    Returns `(residues, feedthrough)` with shapes `(r, q, m)` and `(q, m)`.
    """
    num_outputs = transfer_function.shape[1]
    num_inputs = transfer_function.shape[2]
    model_order = phi0.shape[1] - 1
    transfer_pairs = (
        transfer_function.transpose(1, 2, 0).reshape(-1, transfer_function.shape[0]).T
    )

    def solve_lstsq(V):
        sol, *_ = jnp.linalg.lstsq(phi0, V, rcond=None)
        return sol  # shape (model_order + 1,)

    solutions = jax.vmap(solve_lstsq, in_axes=1)(transfer_pairs)
    solutions = solutions.reshape((num_outputs, num_inputs, model_order + 1))

    feedthrough = solutions[:, :, 0]
    residues = solutions[:, :, 1:].transpose(2, 0, 1)

    return residues, feedthrough


def optimize_order(error_fn, min_order, max_order):
    """Choose a model order by balancing fit error and complexity.

    Golden-section search over the order minimising MSE(order) + lambda * order, with the complexity penalty lambda
    set from the errors at the ends of the range (geometric mean of |C(max - 1) - C(max)| and C(min) - C(max)),
    following Rosenthal & Vick, "Estimating the Optimal Polynomial Order for the Vector Fitting Algorithm", EMC
    Europe 2023, https://doi.org/10.1109/EMCEurope57790.2023.10274284.

    The search assumes this objective has a single valley in [min_order, max_order]. Responses with many fringes or
    resonances have a plateau-then-cliff error curve that violates this below the cliff, so the bounds matter:
    `order_estimation.estimate_order_bounds` estimates them a priori from the Loewner singular values of the data
    (Mayo & Antoulas 2007, https://doi.org/10.1016/j.laa.2007.03.008; Lefteriu & Antoulas 2010,
    https://doi.org/10.1109/TCAD.2009.2034500), and `SParameterElement` uses that estimate when the bounds are unset.

    Parameters
    ----------
    error_fn:
        Callable accepting an integer model order and returning its scalar mean
        squared error.
    min_order:
        Minimum model order to consider.
    max_order:
        Maximum model order to consider.

    Returns
    -------
    int
        The selected model order.
    """
    error_cache = {}

    def cached_error(model_order):
        model_order = int(model_order)
        if model_order not in error_cache:
            error_cache[model_order] = error_fn(model_order)
        return error_cache[model_order]

    C_min = cached_error(min_order)
    C_max = cached_error(max_order)
    C_max_minus_1 = cached_error(max_order - 1)
    lambda_lower = max(abs(C_max_minus_1 - C_max), float(jnp.finfo(float).tiny))
    lambda_upper = max(C_min - C_max, lambda_lower)
    l = math.log10(lambda_lower)
    u = math.log10(lambda_upper)
    complexity_penalty = 10 ** (0.5 * (u + l))

    # Golden-section search minimizing C(order) + complexity_penalty * order
    golden_ratio = (math.sqrt(5) - 1) / 2
    a = min_order
    b = max_order
    c = int(b - golden_ratio * (b - a))
    d = int(a + golden_ratio * (b - a))

    fc = cached_error(c) + complexity_penalty * c
    fd = cached_error(d) + complexity_penalty * d
    while abs(b - a) > 1:
        if fc < fd:  # minimum is in [a, d]
            b, d, fd = d, c, fc
            c = int(b - golden_ratio * (b - a))
            fc = cached_error(c) + complexity_penalty * c
        else:  # minimum is in [c, b]
            a, c, fc = c, d, fd
            d = int(a + golden_ratio * (b - a))
            fd = cached_error(d) + complexity_penalty * d

    best_order = int(round((a + b) / 2))

    return best_order


def _hold_matrices(A, B, T, hold):
    """Exact one-step propagation of `dx/dt = A x + B u(t)` over a sample period `T`.

    `u` is held constant (`hold="zoh"`) or varies linearly (`hold="foh"`) between samples. The matrices come from
    one matrix exponential of an augmented system, as in `scipy.signal.cont2discrete` and `scipy.signal.lsim`.

    Returns `(Ad, Bd)` for `"zoh"`, with `x[k+1] = Ad x[k] + Bd u[k]`, and `(Ad, Gamma1, Gamma2)` for `"foh"`, with
    `x[k+1] = Ad x[k] + (Gamma1 - Gamma2) u[k] + Gamma2 u[k+1]`.
    """
    n, m = B.shape
    dtype = jnp.result_type(A, B, complex)
    A = jnp.asarray(A, dtype=dtype)
    B = jnp.asarray(B, dtype=dtype)
    if hold == "zoh":
        # expm([[A, B], [0, 0]] T) = [[e^{AT}, int_0^T e^{A tau} dtau B], [0, I]]
        block = jnp.zeros((n + m, n + m), dtype=dtype)
        block = block.at[:n, :n].set(A * T).at[:n, n:].set(B * T)
        exponential = jax.scipy.linalg.expm(block)
        return exponential[:n, :n], exponential[:n, n:]
    if hold == "foh":
        # expm([[A T, B T, 0], [0, 0, I], [0, 0, 0]]) = [[e^{AT}, Gamma1, Gamma2], [0, I, I], [0, 0, I]]
        block = jnp.zeros((n + 2 * m, n + 2 * m), dtype=dtype)
        block = block.at[:n, :n].set(A * T).at[:n, n : n + m].set(B * T)
        block = block.at[n : n + m, n + m :].set(jnp.eye(m, dtype=dtype))
        exponential = jax.scipy.linalg.expm(block)
        return exponential[:n, :n], exponential[:n, n : n + m], exponential[:n, n + m :]
    raise ValueError(f"unknown hold {hold!r}; use 'zoh' or 'foh'")


def _state_space(poles, residues, feedthrough):
    """Replicated-pole state-space realization of a pole-residue model.

    Identical in both domains: each pole is replicated once per input, so `A` is diagonal and `B` stacks identities.
    Returns `(A, B, C, D)` with shapes `(r*m, r*m)`, `(r*m, m)`, `(q, r*m)` and `(q, m)`.
    """
    r, q, m = residues.shape

    A = jnp.kron(jnp.diag(poles), jnp.eye(m, dtype=complex))
    B = jnp.tile(jnp.eye(m, dtype=complex), (r, 1))
    # state i*m + j belongs to pole i and input j, and feeds the outputs with residues[i, :, j]
    C = residues.transpose(1, 0, 2).reshape(q, r * m).astype(complex)
    D = feedthrough

    return A, B, C, D
