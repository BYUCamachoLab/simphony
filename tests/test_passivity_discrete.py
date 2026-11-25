"""Tests for passivity assessment and enforcement of z-domain pole-residue models."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import speed_of_light

from simphony.conventions import ENGINEER, PHYSICIST
from simphony.libraries.ideal.s_parameters import optical_s_parameter
from simphony.libraries.siepic import directional_coupler
from simphony.simulation.block_mode import BlockModeSimulationParameters
from simphony.time_domain.vector_fitting.passivity import (
    _coefficients,
    _sigma_max,
    _solve_dual,
    enforce_passivity_discrete,
    passivity_assessment_discrete,
)
from simphony.time_domain.vector_fitting.z_domain import (
    optimize_order_vector_fitting_discrete,
    pole_residue_response_discrete,
    vector_fitting_discrete,
)

WL0 = 1.55e-6
F0 = speed_of_light / WL0
FS = 1e14  # dt = 10 fs
DC_PORTS = ["port_1", "port_2", "port_3", "port_4"]


def fine_sigma_max(poles, residues, feedthrough, sign_convention=PHYSICIST, n=200_000):
    omega = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return _sigma_max(omega, np.asarray(poles), _coefficients(residues, feedthrough), sign_convention).max()


def one_pole(a, p):
    return np.array([p]), np.array([[[a]]], complex), np.zeros((1, 1), complex)


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_assessment_finds_single_pole_peak(sign_convention):
    # |a / (z - p)| peaks at z = p / |p|, i.e. Omega = sign_convention * angle(p), with value |a| / (1 - |p|).
    p = 0.999 * np.exp(1j * 0.7)
    poles, residues, feedthrough = one_pole(0.002, p)
    report = passivity_assessment_discrete(poles, residues, feedthrough, sign_convention=sign_convention)
    assert report.max_singular_value == pytest.approx(0.002 / 0.001, rel=1e-9)
    assert report.max_omega == pytest.approx(sign_convention * 0.7, abs=1e-6)
    assert not report.is_passive
    assert report.violation_omegas.size == 1


def test_assessment_matches_pole_residue_response():
    rng = np.random.default_rng(0)
    poles = 0.9 * np.exp(1j * rng.uniform(-np.pi, np.pi, 6))
    residues = 0.02 * (rng.normal(size=(6, 2, 3)) + 1j * rng.normal(size=(6, 2, 3)))
    feedthrough = 0.1 * (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
    report = passivity_assessment_discrete(poles, residues, feedthrough)
    f = F0 + report.omega * FS / (2 * np.pi)
    H = np.asarray(pole_residue_response_discrete(f, F0, FS, poles, residues, feedthrough, sign_convention=PHYSICIST))
    np.testing.assert_allclose(report.singular_values, np.linalg.svd(H, compute_uv=False)[:, 0], rtol=1e-9)


def test_passive_model_is_unchanged():
    poles, residues, feedthrough = one_pole(0.0005, 0.999 * np.exp(1j * 0.3))
    new_r, new_d, info = enforce_passivity_discrete(poles, residues, feedthrough, np.linspace(-0.1, 0.1, 50))
    assert info["passive"] and info["iterations"] == 0 and info["relative_change"] == 0
    np.testing.assert_array_equal(new_r, residues)
    np.testing.assert_array_equal(new_d, feedthrough)


def test_dual_solve_with_rank_deficient_cut_matrix():
    # More cuts than real coefficients makes M = A A^T rank deficient; rounding can leave its null-space eigenvalues
    # slightly negative, beyond the 1e-12 ridge, where a Cholesky factorization fails (seen on the FMCW 5x1 front
    # end at dt = 2.5 ps: 321 cuts, 170 coefficients). The dual must still be solved.
    rng = np.random.default_rng(1)
    A = rng.normal(size=(60, 10))
    M = A @ A.T
    w, V = np.linalg.eigh(M)
    M -= 1e-9 * np.trace(M) / M.shape[0] * (V[:, :50] @ V[:, :50].T)  # null-space eigenvalues -> slightly < 0
    ridge = 1e-12 * np.trace(M) / M.shape[0]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(M + ridge * np.eye(M.shape[0]))
    lam_star = np.abs(rng.normal(size=60))
    b = -M @ lam_star  # lam_star >= 0 is then an unconstrained (hence constrained) minimizer
    lam = _solve_dual(M, b)

    def objective(x):
        return 0.5 * x @ M @ x + b @ x

    assert np.all(lam >= 0) and np.all(np.isfinite(lam))
    assert objective(lam) <= objective(lam_star) + 1e-8 * abs(objective(lam_star))


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_out_of_band_gain_is_removed_with_small_in_band_change(sign_convention):
    # A passive 2x2 in-band model plus a sharp out-of-band resonance with gain 1.5.
    in_band_poles = 0.95 * np.exp(1j * np.array([-0.05, 0.05]))
    in_band_res = 0.01 * np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]], complex)
    feedthrough = 0.3 * np.eye(2, dtype=complex)
    bad_pole = 0.998 * np.exp(1j * 2.0)
    poles = np.concatenate([in_band_poles, [bad_pole]])
    residues = np.concatenate([in_band_res, 0.003 * np.eye(2, dtype=complex)[None]])
    fit_omega = sign_convention * np.linspace(-0.2, 0.2, 400)

    before = passivity_assessment_discrete(poles, residues, feedthrough, sign_convention=sign_convention)
    assert before.max_singular_value > 1.4
    new_r, new_d, info = enforce_passivity_discrete(
        poles, residues, feedthrough, fit_omega, sign_convention=sign_convention
    )
    assert info["passive"]
    assert fine_sigma_max(poles, new_r, new_d, sign_convention) <= 1.0
    # The fitted band barely moves: the correction goes into the out-of-band residue.
    assert info["relative_change"] < 1e-2
    assert abs(new_r[2, 0, 0]) < abs(residues[2, 0, 0])


def test_enforcement_on_a_real_fit():
    # The SiEPIC directional coupler data has sigma_max = 1.0023 in band, and its z fit inherits that.
    f = np.linspace(speed_of_light / 1.6e-6, speed_of_light / 1.5e-6, 1000)
    sdict = directional_coupler(wl=1e6 * speed_of_light / f)
    S = np.stack([np.stack([np.asarray(sdict[(o, i)]) for i in DC_PORTS], -1) for o in DC_PORTS], -2)
    poles, residues, feedthrough = vector_fitting_discrete(
        16, jnp.asarray(S), jnp.asarray(f), F0, FS, sign_convention=PHYSICIST
    )
    assert not passivity_assessment_discrete(poles, residues, feedthrough).is_passive
    fit_omega = 2 * np.pi * (f - F0) / FS
    new_r, new_d, info = enforce_passivity_discrete(poles, residues, feedthrough, fit_omega)
    assert info["passive"]
    assert fine_sigma_max(poles, new_r, new_d) <= 1.0
    assert info["relative_change"] < 2e-3


def _dc_element(passivity):
    params = BlockModeSimulationParameters(dt=1 / FS, num_time_steps=8, mode_identifiers=("te",))
    model = optical_s_parameter(directional_coupler)
    element = model(
        params,
        sax_settings={},
        vector_fitting_parameters={"model_order": 16, "passivity": passivity},
    )
    return element._state_space(params)  # the fit (and the passivity step) runs here


def test_s_parameter_element_options():
    with pytest.warns(UserWarning, match="not passive"):
        _dc_element("check")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # enforcement succeeds without warning
        _dc_element("enforce")
    with pytest.raises(ValueError, match="passivity"):
        _dc_element("always")


def _dc_data():
    f = np.linspace(speed_of_light / 1.6e-6, speed_of_light / 1.5e-6, 1000)
    sdict = directional_coupler(wl=1e6 * speed_of_light / f)
    S = np.stack([np.stack([np.asarray(sdict[(o, i)]) for i in DC_PORTS], -1) for o in DC_PORTS], -2)
    return jnp.asarray(S), jnp.asarray(f)


def _enforced_mse(S, f, poles, residues, feedthrough):
    new_r, new_d, info = enforce_passivity_discrete(poles, residues, feedthrough, 2 * np.pi * (np.asarray(f) - F0) / FS)
    H = pole_residue_response_discrete(f, F0, FS, poles, new_r, new_d, sign_convention=PHYSICIST)
    return float(np.mean(np.abs(np.asarray(S) - np.asarray(H)) ** 2)), info


def test_order_search_scored_after_enforcement():
    # Scoring candidates after enforcement must not end worse (after enforcement) than the plain search, and the
    # plain search (passivity_enforcement=None) is unchanged by the new option.
    S, f = _dc_data()
    kw = dict(sign_convention=PHYSICIST, use_cache=False)
    plain = optimize_order_vector_fitting_discrete(4, 24, S, f, F0, FS, **kw)
    plain_again = optimize_order_vector_fitting_discrete(4, 24, S, f, F0, FS, passivity_enforcement=None, **kw)
    np.testing.assert_array_equal(np.asarray(plain[0]), np.asarray(plain_again[0]))
    aware = optimize_order_vector_fitting_discrete(4, 24, S, f, F0, FS, passivity_enforcement={}, **kw)
    mse_plain, _ = _enforced_mse(S, f, *plain[:3])
    mse_aware, info = _enforced_mse(S, f, *aware[:3])
    assert info["passive"]
    assert mse_aware <= mse_plain * (1 + 1e-9)


def test_element_passivity_options_and_aware_order():
    params = BlockModeSimulationParameters(dt=1 / FS, num_time_steps=8, mode_identifiers=("te",))
    model = optical_s_parameter(directional_coupler)

    def build(**vf):
        element = model(params, sax_settings={}, vector_fitting_parameters=vf)
        return element._state_space(params)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        build(model_order=16, passivity="enforce", passivity_options={"margin": 1e-3})
        # bounds at or above the a-priori estimate (8 for this coupler), so no low-minimum warning
        build(min_model_order=8, max_model_order=16, passivity="enforce")                    # plain (default)
        build(min_model_order=8, max_model_order=16, passivity="enforce", passivity_aware_order=True)
    with pytest.raises(TypeError):
        build(model_order=16, passivity="enforce", passivity_options={"no_such_option": 1})
