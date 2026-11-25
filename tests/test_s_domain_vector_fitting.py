"""s-domain vector fitting: exact-model recovery, sign conventions, center frequency, state space, order search."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from simphony.conventions import ENGINEER, PHYSICIST
from simphony.time_domain.vector_fitting.s_domain import (
    optimize_order_vector_fitting_continuous,
    pole_residue_response_continuous,
    state_space_continuous,
    state_space_frequency_response_continuous,
    vector_fitting_continuous,
)

F0 = 193.4e12  # optical carrier, Hz
BAND = 400e9  # fitted band, Hz
ORDER = 7  # odd on purpose
FREQUENCY = jnp.asarray(F0 + np.linspace(-BAND / 2, BAND / 2, 400))


def exact_model(sign_convention, seed=0):
    """A stable 2-output x 3-input pole-residue model with resonances inside the band (frequencies in Hz)."""
    rng = np.random.default_rng(seed)
    resonance = rng.uniform(-0.4, 0.4, ORDER) * BAND
    damping = rng.uniform(0.01, 0.05, ORDER) * 2 * np.pi * BAND / 2
    poles = sign_convention * 1j * 2 * np.pi * resonance - damping
    residues = (rng.normal(size=(ORDER, 2, 3)) + 1j * rng.normal(size=(ORDER, 2, 3))) * damping[:, None, None]
    feedthrough = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    return jnp.asarray(poles), jnp.asarray(residues), jnp.asarray(feedthrough)


def matched(reference_poles, poles):
    """Indices pairing `poles` with `reference_poles` (eig returns poles in arbitrary order)."""
    i, j = linear_sum_assignment(np.abs(np.asarray(reference_poles)[:, None] - np.asarray(poles)[None, :]))
    return i, j


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_recovers_an_exact_rectangular_model(sign_convention):
    poles, residues, feedthrough = exact_model(sign_convention)
    H = pole_residue_response_continuous(FREQUENCY, F0, poles, residues, feedthrough, sign_convention)

    p, r, d, info = vector_fitting_continuous(
        ORDER, H, FREQUENCY, F0, sign_convention=sign_convention, max_iterations=20, return_info=True, use_cache=False
    )
    mse = info["mean_squared_error"]

    i, j = matched(poles, p)
    np.testing.assert_allclose(p[j], poles[i], rtol=1e-8)
    np.testing.assert_allclose(r[j], residues[i], rtol=1e-6, atol=1e-6 * np.max(np.abs(residues)))
    np.testing.assert_allclose(d, feedthrough, atol=1e-8)
    assert mse < 1e-20
    assert np.all(np.asarray(p).real < 0)


def test_physicist_fit_is_the_conjugate_of_the_engineer_fit():
    poles, residues, feedthrough = exact_model(ENGINEER)
    H = pole_residue_response_continuous(FREQUENCY, F0, poles, residues, feedthrough, ENGINEER)
    H = H + 1e-3 * jnp.asarray(np.random.default_rng(1).normal(size=H.shape))  # noisy, so the fit is not exact

    p_e, r_e, d_e, info_e = vector_fitting_continuous(
        ORDER, H, FREQUENCY, F0, sign_convention=ENGINEER, return_info=True, use_cache=False
    )
    p_p, r_p, d_p, info_p = vector_fitting_continuous(
        ORDER, jnp.conj(H), FREQUENCY, F0, sign_convention=PHYSICIST, return_info=True, use_cache=False
    )
    mse_e, mse_p = info_e["mean_squared_error"], info_p["mean_squared_error"]

    i, j = matched(jnp.conj(p_e), p_p)
    np.testing.assert_allclose(p_p[j], jnp.conj(p_e)[i], rtol=1e-8)
    np.testing.assert_allclose(r_p[j], jnp.conj(r_e)[i], rtol=1e-6, atol=1e-8 * np.max(np.abs(r_e)))
    np.testing.assert_allclose(d_p, jnp.conj(d_e), atol=1e-8)
    np.testing.assert_allclose(mse_p, mse_e, rtol=1e-6)


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_center_frequency_shifts_the_poles(sign_convention):
    """Poles are relative to the center frequency: moving it by df moves them by -sc j 2 pi df."""
    poles, residues, feedthrough = exact_model(sign_convention)
    H = pole_residue_response_continuous(FREQUENCY, F0, poles, residues, feedthrough, sign_convention)
    shift = 30e9

    p, _, _, info = vector_fitting_continuous(
        ORDER, H, FREQUENCY, F0 + shift, sign_convention=sign_convention, max_iterations=20, return_info=True,
        use_cache=False,
    )
    mse = info["mean_squared_error"]

    expected = poles - sign_convention * 1j * 2 * np.pi * shift
    i, j = matched(expected, p)
    np.testing.assert_allclose(p[j], expected[i], rtol=1e-8)
    assert mse < 1e-20


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_state_space_matches_pole_residue_response(sign_convention):
    poles, residues, feedthrough = exact_model(sign_convention)
    A, B, C, D = state_space_continuous(poles, residues, feedthrough)
    f = FREQUENCY[::7]

    np.testing.assert_allclose(
        state_space_frequency_response_continuous(A, B, C, D, f, F0, sign_convention),
        pole_residue_response_continuous(f, F0, poles, residues, feedthrough, sign_convention),
        rtol=1e-10,
    )


def test_optimize_order_stays_within_bounds():
    poles, residues, feedthrough = exact_model(PHYSICIST)
    H = pole_residue_response_continuous(FREQUENCY, F0, poles, residues, feedthrough)

    p, r, d, info = optimize_order_vector_fitting_continuous(3, 12, H, FREQUENCY, F0, return_info=True, use_cache=False)
    mse = info["mean_squared_error"]

    assert 3 <= p.shape[0] <= 12
    assert r.shape == (p.shape[0], 2, 3) and d.shape == (2, 3)
    assert mse < 1e-6


@pytest.mark.parametrize("interpolation, scipy_interp", [("linear", True), ("zero", False)])
def test_state_space_response_continuous_matches_scipy_lsim(interpolation, scipy_interp):
    from scipy.signal import lsim

    from simphony.time_domain.vector_fitting.s_domain import state_space_response_continuous

    rng = np.random.default_rng(3)
    n, m, q, fs = 6, 2, 3, 1e12
    A = rng.normal(size=(n, n)) * 2e11
    A = A - (np.max(np.linalg.eigvals(A).real) + 1e11) * np.eye(n)
    B, C, D = rng.normal(size=(n, m)) * 1e11, rng.normal(size=(q, n)), rng.normal(size=(q, m))
    u = rng.normal(size=(200, m))
    t = np.arange(200) / fs

    _, y_expected, x_expected = lsim((A, B, C, D), u, t, interp=scipy_interp)
    y, x = state_space_response_continuous(A, B, C, D, u, fs, interpolation=interpolation)

    np.testing.assert_allclose(np.asarray(y).real, y_expected, rtol=1e-8, atol=1e-10 * np.max(np.abs(y_expected)))
    np.testing.assert_allclose(np.asarray(x).real, x_expected, rtol=1e-8, atol=1e-10 * np.max(np.abs(x_expected)))


def test_state_space_response_continuous_single_pole_step():
    from simphony.time_domain.vector_fitting.s_domain import state_space_response_continuous

    # H(s) = r/(s - p) driven by a unit step: y(t) = (r/p)(e^{pt} - 1), exact for either interpolation
    p, r, fs = -2e11 + 5e11j, 3e11 - 1e11j, 1e12
    A, B, C, D = state_space_continuous(jnp.array([p]), jnp.full((1, 1, 1), r), jnp.zeros((1, 1)))
    t = np.arange(100) / fs

    for interpolation in ("linear", "zero"):
        y, _ = state_space_response_continuous(A, B, C, D, jnp.ones((100, 1)), fs, interpolation=interpolation)
        np.testing.assert_allclose(np.asarray(y)[:, 0], (r / p) * (np.exp(p * t) - 1), rtol=1e-10, atol=1e-13)
