"""Continuous-time (s-domain) vector fitting and state-space models.

The basis variable is s = sign_convention * j 2 pi (f - f_c), so `PHYSICIST` (-1) gives s = -j omega and `ENGINEER`
(+1) gives s = +j omega, with omega the baseband angular frequency. Stable poles lie in the left half-plane in both
conventions, and z = e^{sT} links this module to `z_domain` in either convention (see
`z_domain.bilinear_transform` and `z_domain.continuous_to_discrete_state_space`).

Poles and residues are in rad/s, relative to `center_frequency`.
"""

import jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp

from simphony.conventions import ENGINEER, PHYSICIST
from simphony.performance.performance import persistent_cache
from simphony.time_domain.vector_fitting.common import (
    _clear_vector_fitting_memory,
    _empty_history,
    _fit_to_poles,
    _hold_matrices,
    _lstsq_matrices,
    _relocate_poles,
    _state_space,
    _record,
    _vector_fit_info,
    _weight_error,
    optimize_order,
)


def _initial_poles(model_order, frequency, alpha, sign_convention):
    """Complex (CVF) starting poles spread over the band, with a common damping of alpha times the band edge."""
    f = jnp.linspace(jnp.min(frequency), jnp.max(frequency), model_order)
    damping = alpha * 2 * jnp.pi * jnp.max(jnp.abs(frequency))
    poles = sign_convention * 1j * 2 * jnp.pi * f - damping
    return poles


def _phi_matrices(frequency, poles, sign_convention):
    s = sign_convention * 1j * 2 * jnp.pi * frequency
    phi1 = 1 / (s[:, None] - poles[None, :])

    unity_column = jnp.ones((len(s), 1))

    phi0 = jnp.hstack((unity_column, phi1))

    return phi0, phi1


def pole_residue_response_continuous(
    frequency,
    center_frequency,
    poles,
    residues,
    feedthrough,
    sign_convention=PHYSICIST,
):
    """Evaluate a continuous pole-residue transfer function.

    Parameters
    ----------
    frequency:
        Frequency samples, in Hz, where the response is evaluated.
    center_frequency:
        Frequency, in Hz, used as the baseband expansion point.
    poles:
        Continuous poles, in rad/s, with shape `(r,)`.
    residues:
        Residue tensor, in rad/s, with shape `(r, q, m)` for `r` poles, `q`
        outputs, and `m` inputs.
    feedthrough:
        Direct term with shape `(q, m)`.
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.

    Returns
    -------
    jax.Array
        Frequency response with shape `(len(frequency), q, m)`.
    """
    s = sign_convention * 1j * 2 * jnp.pi * (frequency - center_frequency)
    response_shape = (s.shape[0], feedthrough.shape[0], feedthrough.shape[1])
    initial_response = jnp.broadcast_to(feedthrough[None, :, :], response_shape)

    def add_pole(frequency_response, pole_and_residue):
        pole, residue = pole_and_residue
        pole_response = residue[None, :, :] / (s[:, None, None] - pole)
        return frequency_response + pole_response, None

    frequency_response, _ = jax.lax.scan(add_pole, initial_response, (poles, residues))
    return frequency_response


def _mean_squared_error(
    transfer_function,
    frequency,
    center_frequency,
    poles,
    residues,
    feedthrough,
    sign_convention,
):
    fit = pole_residue_response_continuous(
        frequency,
        center_frequency,
        poles,
        residues,
        feedthrough,
        sign_convention=sign_convention,
    )
    error = jnp.mean(jnp.abs(transfer_function - fit) ** 2)

    return error


@persistent_cache
def vector_fitting_continuous(
    model_order,
    transfer_function,
    frequency,
    center_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    alpha=0.01,
    weight_threshold=0.0,
    return_info=False,
):
    """Fit a continuous-time pole-residue model to sampled transfer data.

    This is the core s-domain vector fitting routine. It relocates poles until
    the weighting function converges or `max_iterations` is reached, then solves
    for residues and feedthrough for all input/output pairs. The fit runs in
    frequency normalized to the band edge, so the result does not depend on the
    frequency unit's conditioning.

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
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.
    max_iterations:
        Maximum number of pole-relocation iterations.
    alpha:
        Damping of the initial poles, as a fraction of the band edge
        `2 pi max|f - center_frequency|`.
    weight_threshold:
        Stop when the weight error is at or below this value.
    return_info:
        If True, also return a dict with the fit errors and the error, weight
        error and model (in rad/s) after every iteration (default False).

    Returns
    -------
    tuple
        `(poles, residues, feedthrough)` with shapes `(model_order,)`,
        `(model_order, q, m)` and `(q, m)`. Poles and residues are in rad/s,
        relative to `center_frequency`. With `return_info=True`,
        `(poles, residues, feedthrough, info)`: `info` is a dict with the
        returned model's `"mean_squared_error"`, `"max_abs_error"` and
        `"mean_abs_error"` against `transfer_function`, `"iterations"`,
        `"converged"`, and `"history"` (per-iteration errors, weight errors
        and models; see `vector_fitting.common`).
    """
    baseband_frequency = frequency - center_frequency
    # Fit in normalized frequency (max |f| = 1): with f in Hz the basis columns 1/(s - p) are ~1e-14 next to the
    # unity column and the least-squares problems are numerically singular. Results are rescaled below.
    frequency_scale = jnp.max(jnp.abs(baseband_frequency))
    normalized_frequency = baseband_frequency / frequency_scale

    def poles_not_converged(state):
        _, weight_error, iteration = state

        return (iteration < max_iterations) & (weight_error > weight_threshold)

    def relocate_poles(state):
        previous_poles, _, iteration = state
        phi0, phi1 = _phi_matrices(
            normalized_frequency, previous_poles, sign_convention
        )
        M, B = _lstsq_matrices(model_order, transfer_function, phi0, phi1)
        weight_coeffs, *_ = jnp.linalg.lstsq(M, B)
        weight_error = _weight_error(phi1, weight_coeffs)

        current_poles = _relocate_poles(previous_poles, weight_coeffs)
        # Reflect across the imaginary axis with -conj(p): keeps the pole frequency and flips its damping.
        mask = current_poles.real > 0
        current_poles = jnp.where(mask, -jnp.conj(current_poles), current_poles)

        return (current_poles, weight_error, iteration + 1)

    initial_poles = _initial_poles(
        model_order, normalized_frequency, alpha, sign_convention
    )
    initial_state = (initial_poles, jnp.inf, 0)
    if not return_info:
        final_poles, *_ = jax.lax.while_loop(
            poles_not_converged, relocate_poles, initial_state
        )
    else:
        # the same relocation, also recording each iterate's model (rescaled to rad/s) and errors
        def record(history, k, poles, weight_error):
            phi0, _ = _phi_matrices(normalized_frequency, poles, sign_convention)
            residues, feedthrough = _fit_to_poles(transfer_function, phi0)
            poles, residues = poles * frequency_scale, residues * frequency_scale
            fit = pole_residue_response_continuous(
                frequency, center_frequency, poles, residues, feedthrough, sign_convention=sign_convention
            )
            return _record(history, k, poles, residues, feedthrough, transfer_function, fit, weight_error)

        def relocate_and_record(state):
            poles, weight_error, iteration = relocate_poles(state[:3])
            return (poles, weight_error, iteration, record(state[3], iteration, poles, weight_error))

        history = record(_empty_history(max_iterations, initial_poles, transfer_function), 0, initial_poles, jnp.nan)
        final_poles, final_weight_error, iterations, history = jax.lax.while_loop(
            lambda state: poles_not_converged(state[:3]), relocate_and_record, (*initial_state, history)
        )
    phi0, _ = _phi_matrices(normalized_frequency, final_poles, sign_convention)
    residues, feedthrough = _fit_to_poles(transfer_function, phi0)

    # r/(s/c - p) = (c r)/(s - c p): poles and residues scale with c, the feedthrough does not
    final_poles = final_poles * frequency_scale
    residues = residues * frequency_scale

    if return_info:
        return final_poles, residues, feedthrough, _vector_fit_info(
            history, iterations, final_weight_error, max_iterations, weight_threshold
        )
    return final_poles, residues, feedthrough


@persistent_cache
def optimize_order_vector_fitting_continuous(
    min_order,
    max_order,
    transfer_function,
    frequency,
    center_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    alpha=0.01,
    weight_threshold=0.0,
    return_info=False,
):
    """Fit a continuous pole-residue model while selecting the model order.

    This wraps `vector_fitting_continuous` with `optimize_order`, searching
    between `min_order` and `max_order`.

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
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.
    max_iterations:
        Maximum pole-relocation iterations for each attempted order.
    alpha:
        Initial pole damping used by `vector_fitting_continuous`.
    weight_threshold:
        Pole-relocation convergence threshold.
    return_info:
        If True, also return the selected fit's info dict (as
        `vector_fitting_*` returns it) with an `"order_search"` entry: the
        bounds, the complexity penalty, every order evaluated with its score,
        and the selected order (default False).

    Returns
    -------
    tuple
        `(poles, residues, feedthrough)` for the selected order, or
        `(poles, residues, feedthrough, info)` with `return_info=True`.
    """

    def fit_order(model_order, return_info=False):
        return vector_fitting_continuous(
            model_order,
            transfer_function,
            frequency,
            center_frequency,
            sign_convention=sign_convention,
            max_iterations=max_iterations,
            alpha=alpha,
            weight_threshold=weight_threshold,
            return_info=return_info,
            use_cache=False,
        )

    def error_fn(model_order):
        poles = residues = feedthrough = mean_squared_error = None
        try:
            poles, residues, feedthrough = fit_order(model_order)
            mean_squared_error = _mean_squared_error(
                transfer_function, frequency, center_frequency, poles, residues, feedthrough, sign_convention
            )
            return float(jax.device_get(mean_squared_error))
        finally:
            del poles, residues, feedthrough, mean_squared_error
            _clear_vector_fitting_memory()

    if not return_info:
        best_order = optimize_order(error_fn, min_order, max_order)
        try:
            return fit_order(best_order)
        except Exception:
            _clear_vector_fitting_memory()
            raise

    best_order, search = optimize_order(error_fn, min_order, max_order, return_info=True)
    try:
        poles, residues, feedthrough, info = fit_order(best_order, return_info=True)
    except Exception:
        _clear_vector_fitting_memory()
        raise
    info["order_search"] = dict(min_order=int(min_order), max_order=int(max_order), selected_order=best_order, **search)
    return poles, residues, feedthrough, info


def state_space_continuous(poles, residues, feedthrough):
    """Convert a pole-residue model into a continuous state-space realization.

    The realization replicates each pole once per input channel. It is simple
    and deterministic, but it is not guaranteed to be minimal.

    Parameters
    ----------
    poles:
        Continuous poles with shape `(r,)`.
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


def state_space_frequency_response_continuous(
    A, B, C, D, f, f_center, sign_convention=PHYSICIST
):
    """Compute the frequency response of a continuous state-space system.

    Evaluates C (sI - A)^{-1} B + D at s = sign_convention * j 2 pi (f - f_center).

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
    sign_convention:
        Complex exponential convention, usually `PHYSICIST` or `ENGINEER`.

    Returns
    -------
    jax.Array
        Frequency response with shape `(len(f), q, m)`.
    """
    s = sign_convention * 1j * 2 * jnp.pi * (jnp.ravel(f) - f_center)
    eye = jnp.eye(A.shape[0], dtype=complex)
    # (sI - A)^{-1} B for every frequency at once: (n_freq, n, m)
    resolvent_B = jnp.linalg.solve(s[:, None, None] * eye - A, jnp.broadcast_to(B, (s.shape[0], *B.shape)))
    return C @ resolvent_B + D


@jax.jit
def _state_space_response_continuous_zoh(Ad, Bd, C, D, u, x0):
    def step(x, u_k):
        y_k = C @ x + D @ u_k
        return Ad @ x + Bd @ u_k, (y_k, x)

    _, (yout, xout) = jax.lax.scan(step, x0, u)
    return yout, xout


@jax.jit
def _state_space_response_continuous_foh(Ad, Gamma1, Gamma2, C, D, u, x0):
    # the input ramps from u[k] to u[k+1] over each step; the last sample is held, since there is no u[T]
    u_next = jnp.concatenate([u[1:], u[-1:]], axis=0)
    B_current = Gamma1 - Gamma2

    def step(x, inputs):
        u_k, u_k_next = inputs
        y_k = C @ x + D @ u_k
        return Ad @ x + B_current @ u_k + Gamma2 @ u_k_next, (y_k, x)

    _, (yout, xout) = jax.lax.scan(step, x0, (u, u_next))
    return yout, xout


def state_space_response_continuous(A, B, C, D, u, sampling_frequency, x0=None, interpolation="linear"):
    """Simulate a continuous state-space model, `dx/dt = A x + B u(t)`, `y = C x + D u`, over sampled input.

    This is the continuous-time counterpart of `z_domain.state_space_response_discrete`. The differential
    equation is solved exactly between samples for an input that is piecewise linear (`interpolation="linear"`,
    as in `scipy.signal.lsim`) or piecewise constant (`"zero"`). No discrete-time approximation of the model is
    made: `sampling_frequency` sets only the spacing of the input and output samples. The propagation matrices
    come from one matrix exponential, so the cost is a dense `expm` of size `n + 2m` plus one step per sample.

    States evolve as e^{p t} with p a pole of `state_space_continuous`; in the `PHYSICIST` convention
    (s = -j omega) a baseband frequency offset f corresponds to the envelope e^{-j 2 pi f t}.

    Parameters
    ----------
    A, B, C, D:
        Continuous state-space matrices, with A in rad/s.
    u:
        Input samples with shape `(T, m)`, taken at t_k = k / sampling_frequency.
    sampling_frequency:
        Sample rate of `u` and of the returned output, in Hz.
    x0:
        Optional initial state with shape `(n,)`. Defaults to zeros.
    interpolation:
        `"linear"` (first-order hold) or `"zero"` (zero-order hold) between input samples.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Output samples `y(t_k)` with shape `(T, q)` and states `x(t_k)` with shape `(T, n)`.
    """
    T = 1 / sampling_frequency
    dtype = jnp.result_type(A, B, C, D, u, complex)
    u = jnp.asarray(u, dtype=dtype)
    C = jnp.asarray(C, dtype=dtype)
    D = jnp.asarray(D, dtype=dtype)
    if x0 is None:
        x0 = jnp.zeros((A.shape[0],), dtype=dtype)
    x0 = jnp.asarray(x0, dtype=dtype)

    if interpolation == "zero":
        Ad, Bd = _hold_matrices(A, B, T, "zoh")
        return _state_space_response_continuous_zoh(Ad, Bd, C, D, u, x0)
    if interpolation == "linear":
        Ad, Gamma1, Gamma2 = _hold_matrices(A, B, T, "foh")
        return _state_space_response_continuous_foh(Ad, Gamma1, Gamma2, C, D, u, x0)
    raise ValueError(f"unknown interpolation {interpolation!r}; use 'linear' or 'zero'")
