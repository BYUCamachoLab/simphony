"""Tests for the `fitter="s_bilinear"` option of `SParameterElement`: a continuous s-domain vector fit at
baseband, mapped to discrete time with the bilinear transform."""

import jax.numpy as jnp
import numpy as np
from scipy.constants import speed_of_light

from simphony.libraries.ideal.s_parameters import optical_s_parameter
from simphony.simulation.sample_mode import SampleModeSimulationParameters
from simphony.conventions import PHYSICIST
from simphony.time_domain.vector_fitting.s_domain import (
    pole_residue_response_continuous,
    vector_fitting_continuous,
)

WL0 = 1.55e-6
F0 = speed_of_light / WL0
DT = 1e-12
TAU = 0.3e-12        # a fraction of a sample: the case where s fits beat z fits
SPAN = 0.25 / DT     # fitted band, Hz


def delay_line(wl=1.55, tau=TAU, loss=0.9):
    """Lossy delay in the physicist's convention: S21 = loss * exp(+j 2 pi f tau)."""
    f = speed_of_light / (jnp.asarray(wl) * 1e-6)
    s21 = loss * jnp.exp(2j * jnp.pi * f * tau)
    return {("o1", "o2"): s21, ("o2", "o1"): s21}


def fit_settings(fitter, order=6):
    return {
        "vector_fitting_parameters": {
            "model_order": order,
            "num_frequency_samples": 400,
            "center_wavelength": WL0,
            "spectral_range": (speed_of_light / (F0 + SPAN / 2), speed_of_light / (F0 - SPAN / 2)),
            "fitter": fitter,
        },
        "port_directionality": {"o1": "input", "o2": "output"},
    }


def state_space(fitter):
    params = SampleModeSimulationParameters(
        dt=DT, num_time_steps=4, mode_identifiers=("te",), optical_baseband_wavelengths=jnp.array([WL0])
    )
    element = optical_s_parameter(delay_line)(params, **fit_settings(fitter))
    ss = element._state_space(params)
    return [np.asarray(ss[k]) for k in "ABCD"]


def response(A, B, C, D, baseband):
    """C (zI - A)^-1 B + D on the physicist's unit circle, z = exp(-j 2 pi f dt)."""
    z = np.exp(-2j * np.pi * baseband * DT)
    eye = np.eye(A.shape[0])
    return np.array([C @ np.linalg.solve(zk * eye - A, B) + D for zk in z])


def test_bilinear_model_is_the_warped_continuous_fit():
    """H_discrete(f) = H_continuous(f_a), with f_a = tan(pi f T) / (pi T), in simphony's convention."""
    A, B, C, D = state_space("s_bilinear")
    f = np.linspace(-SPAN / 2, SPAN / 2, 101)
    # the element's own frequency grid, built from the wavelength range: an order-6 fit of this near-linear delay is
    # over-parameterized, so a grid perturbed by rounding (~0.03 Hz) can land on different, equally good poles
    spectral_range = fit_settings("s_bilinear")["vector_fitting_parameters"]["spectral_range"]
    frequency = jnp.linspace(speed_of_light / max(spectral_range), speed_of_light / min(spectral_range), 400)
    data = jnp.asarray(delay_line(wl=1e6 * speed_of_light / frequency)[("o1", "o2")])[:, None, None]
    p, r, d, _ = vector_fitting_continuous(6, data, frequency, F0, sign_convention=PHYSICIST, use_cache=False)
    warped = np.tan(np.pi * f * DT) / (np.pi * DT)
    continuous = np.asarray(
        pole_residue_response_continuous(jnp.asarray(F0 + warped), F0, p, r, d, sign_convention=PHYSICIST)
    )
    np.testing.assert_allclose(response(A, B, C, D, f), continuous, atol=1e-10)


def test_bilinear_model_is_exact_at_the_carrier_and_stable():
    A, B, C, D = state_space("s_bilinear")
    exact = complex(np.asarray(delay_line(wl=WL0 * 1e6)[("o1", "o2")]))
    assert abs(response(A, B, C, D, np.array([0.0]))[0, 0, 0] - exact) < 1e-8
    assert np.max(np.abs(np.diag(A))) < 1


def test_bilinear_error_grows_away_from_the_carrier_while_z_does_not():
    f = np.array([0.0, 0.1 / DT])
    exact = np.asarray(delay_line(wl=1e6 * speed_of_light / (F0 + f))[("o1", "o2")])
    errors = {fitter: np.abs(response(*state_space(fitter), f)[:, 0, 0] - exact) for fitter in ("z", "s_bilinear")}
    assert errors["s_bilinear"][1] > 10 * errors["s_bilinear"][0]
    assert errors["s_bilinear"][1] > errors["z"][1]
