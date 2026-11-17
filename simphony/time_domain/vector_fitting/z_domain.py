"""Discrete-time (z-domain) vector fitting and state-space models.

The basis variable is z = exp(sign_convention * j 2 pi (f - f_c) / f_s), so `PHYSICIST` (-1) gives z = e^{-j Omega}
and `ENGINEER` (+1) gives z = e^{+j Omega}. Stable poles lie inside the unit circle in both conventions. The
continuous-time counterpart is `s_domain`.
"""

import jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

from simphony.conventions import ENGINEER, PHYSICIST
from simphony.performance.performance import persistent_cache
from simphony.time_domain.vector_fitting.common import (
    _clear_vector_fitting_memory,
    _fit_to_poles,
    _hold_matrices,
    _lstsq_matrices,
    _relocate_poles,
    _state_space,
    _weight_error,
    optimize_order,
)


# @jax.jit
def _initial_poles(model_order, frequency, sampling_frequency, gamma, sign_convention):
    f = jnp.linspace(jnp.min(frequency), jnp.max(frequency), model_order)
    poles = gamma * jnp.exp(sign_convention * 1j * 2 * jnp.pi * f / sampling_frequency)
    return poles


# @jax.jit
def _phi_matrices(frequency, sampling_frequency, poles, sign_convention):
    z = jnp.exp(sign_convention * 1j * 2 * jnp.pi * frequency / sampling_frequency)
    phi1 = 1 / (z[:, None] - poles[None, :])

    unity_column = jnp.ones((len(z), 1))

    phi0 = jnp.hstack((unity_column, phi1))

    return phi0, phi1


# @jax.jit
def pole_residue_response_discrete(
    frequency,
    center_frequency,
    sampling_frequency,
    poles,
    residues,
    feedthrough,
    sign_convention=PHYSICIST,
):
    """Evaluate a discrete pole-residue transfer function.

    Parameters
    ----------
    frequency:
        Frequency samples, in Hz, where the response is evaluated.
    center_frequency:
        Frequency, in Hz, used as the baseband expansion point.
    sampling_frequency:
        Discrete-time sampling frequency, in Hz.
    poles:
        Discrete poles with shape `(r,)`.
    residues:
        Residue tensor with shape `(r, q, m)` for `r` poles, `q` outputs, and
        `m` inputs.
    feedthrough:
        Direct term with shape `(q, m)`.
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.

    Returns
    -------
    jax.Array
        Frequency response with shape `(len(frequency), q, m)`.
    """
    z = jnp.exp(
        sign_convention
        * 1j
        * 2
        * jnp.pi
        * (frequency - center_frequency)
        / sampling_frequency
    )
    response_shape = (z.shape[0], feedthrough.shape[0], feedthrough.shape[1])
    initial_response = jnp.broadcast_to(feedthrough[None, :, :], response_shape)

    def add_pole(frequency_response, pole_and_residue):
        pole, residue = pole_and_residue
        pole_response = residue[None, :, :] / (z[:, None, None] - pole)
        return frequency_response + pole_response, None

    frequency_response, _ = jax.lax.scan(add_pole, initial_response, (poles, residues))
    return frequency_response


# @jax.jit
def _mean_squared_error(
    transfer_function,
    frequency,
    center_frequency,
    sampling_frequency,
    poles,
    residues,
    feedthrough,
    sign_convention,
):
    fit = pole_residue_response_discrete(
        frequency,
        center_frequency,
        sampling_frequency,
        poles,
        residues,
        feedthrough,
        sign_convention=sign_convention,
    )
    error = jnp.mean(jnp.abs(transfer_function - fit) ** 2)

    return error


@persistent_cache
def vector_fitting_discrete(
    model_order,
    transfer_function,
    frequency,
    center_frequency,
    sampling_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    gamma=0.95,
    weight_threshold=0.0,
):
    """Fit a discrete-time pole-residue model to sampled transfer data.

    This is the core z-domain vector fitting routine. It relocates poles until
    the weighting function converges or `max_iterations` is reached, then solves
    for residues and feedthrough for all input/output pairs.

    Parameters
    ----------
    model_order:
        Number of poles to fit.
    transfer_function:
        Target response with shape `(num_frequency, q, m)`.
    frequency:
        Frequency samples, in Hz, matching the first axis of
        `transfer_function`.
    center_frequency:
        Center frequency, in Hz, used for basebanding.
    sampling_frequency:
        Discrete-time sampling frequency, in Hz.
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.
    max_iterations:
        Maximum number of pole-relocation iterations.
    gamma:
        Radius used for the initial pole placement inside the unit circle.
    weight_threshold:
        Stop when the weight error is at or below this value.

    Returns
    -------
    tuple
        `(poles, residues, feedthrough, mean_squared_error)` where `poles` has
        shape `(model_order,)`, `residues` has shape `(model_order, q, m)`,
        `feedthrough` has shape `(q, m)`, and `mean_squared_error` is the fit
        error against `transfer_function`.
    """
    baseband_frequency = frequency - center_frequency

    def poles_not_converged(state):
        _, weight_error, iteration = state

        return (iteration < max_iterations) & (weight_error > weight_threshold)

    def relocate_poles(state):
        previous_poles, _, iteration = state
        phi0, phi1 = _phi_matrices(
            baseband_frequency, sampling_frequency, previous_poles, sign_convention
        )
        M, B = _lstsq_matrices(model_order, transfer_function, phi0, phi1)
        weight_coeffs, *_ = jnp.linalg.lstsq(M, B)
        weight_error = _weight_error(phi1, weight_coeffs)

        current_poles = _relocate_poles(previous_poles, weight_coeffs)
        # Reflect across the unit circle with 1/conj(p): keeps the pole angle (its frequency) and
        # inverts its radius. 1/p would also negate the angle, moving complex baseband poles to -f.
        mask = jnp.abs(current_poles) > 1
        current_poles = jnp.where(mask, 1 / jnp.conj(current_poles), current_poles)

        return (current_poles, weight_error, iteration + 1)

    initial_poles = _initial_poles(
        model_order, baseband_frequency, sampling_frequency, gamma, sign_convention
    )
    initial_state = (initial_poles, jnp.inf, 0)
    final_poles, *_ = jax.lax.while_loop(
        poles_not_converged, relocate_poles, initial_state
    )
    phi0, _ = _phi_matrices(
        baseband_frequency, sampling_frequency, final_poles, sign_convention
    )
    residues, feedthrough = _fit_to_poles(transfer_function, phi0)

    error = _mean_squared_error(
        transfer_function,
        frequency,
        center_frequency,
        sampling_frequency,
        final_poles,
        residues,
        feedthrough,
        sign_convention,
    )

    return final_poles, residues, feedthrough, error


# TODO: Cache the model order, not the model itself to save space
@persistent_cache
def optimize_order_vector_fitting_discrete(
    min_order,
    max_order,
    transfer_function,
    frequency,
    center_frequency,
    sampling_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    gamma=0.95,
    weight_threshold=0.0,
):
    """Fit a discrete pole-residue model while selecting the model order.

    This wraps `vector_fitting_discrete` with `optimize_order`, searching between
    `min_order` and `max_order`.

    Parameters
    ----------
    min_order:
        Smallest number of poles to consider.
    max_order:
        Largest number of poles to consider.
    transfer_function:
        Target response with shape `(num_frequency, q, m)`.
    frequency:
        Frequency samples, in Hz.
    center_frequency:
        Center frequency, in Hz, used for basebanding.
    sampling_frequency:
        Discrete-time sampling frequency, in Hz.
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.
    max_iterations:
        Maximum pole-relocation iterations for each attempted order.
    gamma:
        Initial pole radius used by `vector_fitting_discrete`.
    weight_threshold:
        Pole-relocation convergence threshold.

    Returns
    -------
    tuple
        `(poles, residues, feedthrough, mean_squared_error)` for the selected
        order.
    """

    def fit_order(model_order):
        return vector_fitting_discrete(
            model_order,
            transfer_function,
            frequency,
            center_frequency,
            sampling_frequency,
            sign_convention=sign_convention,
            max_iterations=max_iterations,
            gamma=gamma,
            weight_threshold=weight_threshold,
            use_cache=False,
        )

    def error_fn(model_order):
        poles = residues = feedthrough = mean_squared_error = None
        try:
            poles, residues, feedthrough, mean_squared_error = fit_order(model_order)
            return float(jax.device_get(mean_squared_error))
        finally:
            del poles, residues, feedthrough, mean_squared_error
            _clear_vector_fitting_memory()

    best_order = optimize_order(error_fn, min_order, max_order)
    try:
        poles, residues, feedthrough, mean_squared_error = fit_order(best_order)
    except Exception:
        _clear_vector_fitting_memory()
        raise

    return poles, residues, feedthrough, mean_squared_error


def state_space_discrete(poles, residues, feedthrough):
    """Convert a pole-residue model into a discrete state-space realization.

    The realization replicates each pole once per input channel. It is simple
    and deterministic, but it is not guaranteed to be minimal.

    Parameters
    ----------
    poles:
        Discrete poles with shape `(r,)`.
    residues:
        Residue tensor with shape `(r, q, m)`.
    feedthrough:
        Direct term with shape `(q, m)`.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        `(A, B, C, D)` with shapes `(r*m, r*m)`, `(r*m, m)`, `(q, r*m)`, and
        `(q, m)`.
    """
    return _state_space(poles, residues, feedthrough)


def bilinear_transform(poles, residues, feedthrough, sampling_frequency):
    """Map a continuous pole-residue model to discrete time with the bilinear transform.

    s = (2/T)(z - 1)/(z + 1) with T = 1/sampling_frequency and no prewarping, so H(s = 0) maps exactly to
    H(z = 1) (the center frequency) and frequencies are warped as omega_digital = (2/T) arctan(omega T / 2).

    The mapping approximates z = e^{sT}. Since `s_domain` uses s = sc j omega and `z_domain` uses
    z = e^{sc j omega T} for the same sign convention sc, it needs no convention argument. A `PHYSICIST`
    continuous fit becomes a `PHYSICIST` discrete model, ready for `state_space_discrete`. Unlike
    `continuous_to_discrete_state_space`, the result keeps the replicated-pole structure that
    `state_space_response_discrete_optimized` needs.

    Each term r/(s - p) becomes r T/(1 - p T/2)^2 / (z - z_p) + r T/(2(1 - p T/2)), with
    z_p = (1 + p T/2)/(1 - p T/2). Left-half-plane poles land inside the unit circle.

    Parameters
    ----------
    poles:
        Continuous poles, in rad/s, with shape `(r,)`.
    residues:
        Residue tensor, in rad/s, with shape `(r, q, m)`.
    feedthrough:
        Direct term with shape `(q, m)`.
    sampling_frequency:
        Discrete-time sampling frequency, in Hz.

    Returns
    -------
    tuple
        `(poles, residues, feedthrough)` of the discrete model, in the shapes
        of the inputs.
    """
    T = 1 / sampling_frequency
    denominator = 1 - poles * T / 2
    discrete_poles = (1 + poles * T / 2) / denominator
    discrete_residues = residues * (T / denominator**2)[:, None, None]
    discrete_feedthrough = feedthrough + jnp.sum(
        residues * (T / (2 * denominator))[:, None, None], axis=0
    )
    return discrete_poles, discrete_residues, discrete_feedthrough


def continuous_to_discrete_state_space(A, B, C, D, sampling_frequency, method="bilinear"):
    """Discretize a continuous state-space model at `sampling_frequency`.

    The same conversions as `scipy.signal.cont2discrete`, for complex (baseband) models. With T = 1 /
    sampling_frequency:

    - `"bilinear"` (Tustin, no prewarping): s = (2/T)(z - 1)/(z + 1). With F = (I - A T/2)^{-1},
      Ad = F (I + A T/2), Bd = F B T, Cd = C F and Dd = D + C F B T/2. The transfer function equals that of
      `bilinear_transform` applied to the pole-residue model.
    - `"zoh"`: exact for an input held constant between samples. Ad = e^{AT}, Bd = int_0^T e^{A tau} dtau B,
      Cd = C, Dd = D.
    - `"foh"`: exact for an input that varies linearly between samples (the `scipy.signal.lsim` default). With
      Gamma1 and Gamma2 the first-order-hold integrals, Bd = Gamma1 + Ad Gamma2 - Gamma2, Cd = C and
      Dd = D + C Gamma2.

    Every method maps the left half-plane inside the unit circle, and a model in a given sign convention stays in
    that convention (z = e^{sT}). The result works with `state_space_response_discrete` and
    `state_space_frequency_response_discrete`. Its B is not a stack of identities, so
    `state_space_response_discrete_optimized` does not apply; to keep that structure, discretize the
    pole-residue model with `bilinear_transform` and realize it with `state_space_discrete`.

    Parameters
    ----------
    A, B, C, D:
        Continuous state-space matrices, with A in rad/s (for example from `s_domain.state_space_continuous`).
    sampling_frequency:
        Discrete-time sampling frequency, in Hz.
    method:
        `"bilinear"`, `"zoh"` or `"foh"`.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        `(Ad, Bd, Cd, Dd)`, in the shapes of `(A, B, C, D)`.
    """
    T = 1 / sampling_frequency
    dtype = jnp.result_type(A, B, C, D, complex)
    A, B, C, D = (jnp.asarray(matrix, dtype=dtype) for matrix in (A, B, C, D))
    n = A.shape[0]

    if method == "bilinear":
        eye = jnp.eye(n, dtype=dtype)
        F = jnp.linalg.inv(eye - A * T / 2)
        Ad = F @ (eye + A * T / 2)
        Bd = F @ B * T
        Cd = C @ F
        Dd = D + Cd @ B * (T / 2)
        return Ad, Bd, Cd, Dd
    if method == "zoh":
        Ad, Bd = _hold_matrices(A, B, T, "zoh")
        return Ad, Bd, C, D
    if method == "foh":
        Ad, Gamma1, Gamma2 = _hold_matrices(A, B, T, "foh")
        return Ad, Gamma1 + Ad @ Gamma2 - Gamma2, C, D + C @ Gamma2
    raise ValueError(f"unknown method {method!r}; use 'bilinear', 'zoh' or 'foh'")


@jax.jit
def _state_space_response_discrete(A, B, C, D, u, x0):
    def step(x, u_k):
        y_k = C @ x + D @ u_k
        x_next = A @ x + B @ u_k
        return x_next, (y_k, x_next)

    _, (yout, xout) = jax.lax.scan(step, x0, u)
    return yout, xout


def state_space_response_discrete(A, B, C, D, u, x0=None):
    """Simulate a discrete state-space model over an input sequence.

    Parameters
    ----------
    A, B, C, D:
        State-space matrices.
    u:
        Input samples with shape `(T, m)`.
    x0:
        Optional initial state with shape `(n,)`. Defaults to zeros.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Output samples `y` with shape `(T, q)` and state history with shape
        `(T, n)`.
    """
    if x0 is None:
        x0 = jnp.zeros((A.shape[0],), dtype=A.dtype)

    return _state_space_response_discrete(A, B, C, D, u, x0)


def state_space_has_optimized_structure(A, B):
    """Return true when ABCD matrices match `state_space_discrete`
    structure."""
    M = A.shape[0]
    m = B.shape[1]
    if A.shape != (M, M) or B.shape[0] != M or m == 0 or M % m != 0:
        return False

    r = M // m
    expected_B = jnp.tile(jnp.eye(m, dtype=B.dtype), (r, 1))
    diagonal_A = jnp.diag(jnp.diag(A))

    return bool(jnp.allclose(B, expected_B)) and bool(jnp.allclose(A, diagonal_A))


def state_space_discrete_optimized_terms(A, B, C, check_structure=True):
    """Precompute the terms used by optimized structured state-space
    updates."""
    if check_structure and not state_space_has_optimized_structure(A, B):
        raise ValueError(
            "ABCD matrices do not match the optimized state-space structure"
        )

    A_diag = jnp.diag(A)
    return A_diag, C


@jax.jit
def state_space_step_discrete_optimized(A_diag, C, D, update_constant, u, x):
    """Run one batched sample update for the optimized pole-residue
    realization.

    `update_constant` may be any per-wavelength multiplier for the state
    update, not only a unit-magnitude phase.
    """
    r = A_diag.shape[0] // u.shape[1]
    y = x @ C.T + u @ D.T
    x_next = update_constant[:, None] * (A_diag[None, :] * x + jnp.tile(u, (1, r)))
    return y, x_next


@jax.jit
def _state_space_response_discrete_optimized(
    A_diag,
    C,
    D,
    update_constant,
    u,
    x0,
):
    def step(x, u_k):
        y_k, x_next = state_space_step_discrete_optimized(
            A_diag,
            C,
            D,
            update_constant,
            u_k,
            x,
        )
        return x_next, y_k

    x_final, yout = jax.lax.scan(step, x0, u)
    return yout, x_final


def state_space_response_discrete_optimized(
    A,
    B,
    C,
    D,
    update_constant,
    u,
    x0=None,
):
    """Simulate a structured state-space model for several wavelengths at once.

    Parameters
    ----------
    A, B, C, D:
        State-space matrices generated by `state_space_discrete`.
    update_constant:
        Per-wavelength state-update multipliers with shape `(L,)`.
    u:
        Input samples with shape `(T, L, m)`.
    x0:
        Optional initial state with shape `(L, n)`. Defaults to zeros.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Output samples with shape `(T, L, q)` and the final state with shape
        `(L, n)`.
    """
    if u.ndim != 3:
        raise ValueError("u must have shape (time_steps, wavelengths, inputs)")

    if x0 is None:
        x0 = jnp.zeros((u.shape[1], A.shape[0]), dtype=A.dtype)

    A_diag, C = state_space_discrete_optimized_terms(
        A,
        B,
        C,
        check_structure=False,
    )

    return _state_space_response_discrete_optimized(
        A_diag,
        C,
        D,
        update_constant,
        u,
        x0,
    )


def state_space_frequency_response_discrete(
    A, B, C, D, f, f_center, dt, sign_convention=PHYSICIST
):
    """Compute the frequency response of a discrete state-space system.

    Evaluates C (zI - A)^{-1} B + D at z = exp(sign_convention * j 2 pi (f - f_center) dt).

    Parameters
    ----------
    A:  jnp.ndarray, shape (n, n)
        State matrix
    B:  jnp.ndarray, shape (n, m)
        Input matrix
    C:  jnp.ndarray, shape (q, n)
        Output matrix
    D:  jnp.ndarray, shape (q, m)
        Feedthrough matrix
    f:
        Frequency samples, in Hz.
    f_center:
        Center frequency, in Hz, used for basebanding.
    dt:
        Time step, in seconds.
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.

    Returns
    -------
    jax.Array
        Frequency response with shape `(len(f), q, m)`.
    """
    z = jnp.exp(sign_convention * 1j * 2 * jnp.pi * (jnp.ravel(f) - f_center) * dt)
    eye = jnp.eye(A.shape[0], dtype=complex)
    # (zI - A)^{-1} B for every frequency at once: (n_freq, n, m)
    resolvent_B = jnp.linalg.solve(z[:, None, None] * eye - A, jnp.broadcast_to(B, (z.shape[0], *B.shape)))
    return C @ resolvent_B + D
