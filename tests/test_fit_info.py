"""Fitter outputs: `(poles, residues, feedthrough)` by default, plus an info dict with `return_info=True` that
describes the fit (final errors, per-iteration history, order search) without changing it."""

import jax.numpy as jnp
import numpy as np
import pytest

from simphony.conventions import PHYSICIST
from simphony.time_domain.vector_fitting.s_domain import (
    optimize_order_vector_fitting_continuous,
    pole_residue_response_continuous,
    vector_fitting_continuous,
)
from simphony.time_domain.vector_fitting.z_domain import (
    optimize_order_vector_fitting_discrete,
    pole_residue_response_discrete,
    vector_fitting_discrete,
)

F0 = 193.4e12
FS = 1e12
F = jnp.asarray(F0 + np.linspace(-0.4, 0.4, 301) * FS)


def _noise(rng, shape):
    return 1e-4 * jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))   # finite fit errors


def discrete_data(order=6, seed=0):
    rng = np.random.default_rng(seed)
    poles = jnp.asarray(rng.uniform(0.6, 0.9, order) * np.exp(1j * rng.uniform(-2.4, 2.4, order)))
    residues = jnp.asarray((rng.normal(size=(order, 2, 2)) + 1j * rng.normal(size=(order, 2, 2))) * 0.05)
    feedthrough = jnp.asarray(rng.normal(size=(2, 2)) * 0.1 + 0j)
    H = pole_residue_response_discrete(F, F0, FS, poles, residues, feedthrough, sign_convention=PHYSICIST)
    return H + _noise(rng, H.shape)


def continuous_data(order=6, seed=1):
    rng = np.random.default_rng(seed)
    poles = jnp.asarray(-rng.uniform(0.02, 0.1, order) * FS * 2 * np.pi + 2j * np.pi * rng.uniform(-0.35, 0.35, order) * FS)
    residues = jnp.asarray((rng.normal(size=(order, 2, 2)) + 1j * rng.normal(size=(order, 2, 2))) * 0.05 * FS)
    feedthrough = jnp.asarray(rng.normal(size=(2, 2)) * 0.1 + 0j)
    H = pole_residue_response_continuous(F, F0, poles, residues, feedthrough, sign_convention=PHYSICIST)
    return H + _noise(rng, H.shape)


def z_response(p, r, d):
    return pole_residue_response_discrete(F, F0, FS, p, r, d, sign_convention=PHYSICIST)


def s_response(p, r, d):
    return pole_residue_response_continuous(F, F0, p, r, d, sign_convention=PHYSICIST)


def check_info(info, model, data, response):
    poles, residues, feedthrough = model
    n = info["iterations"]
    h = info["history"]
    for key in ("max_abs_error", "mean_abs_error", "mean_squared_error", "weight_error"):
        assert h[key].shape == (n + 1,)
    assert h["poles"].shape == (n + 1, poles.shape[0])
    assert h["residues"].shape == (n + 1, *residues.shape)
    assert h["feedthrough"].shape == (n + 1, *feedthrough.shape)
    assert np.isnan(h["weight_error"][0]) and np.all(np.isfinite(h["weight_error"][1:]))
    # the final errors are those of the returned model, which is the last iterate
    e = np.abs(np.asarray(data) - np.asarray(response(*model)))
    np.testing.assert_allclose(info["max_abs_error"], e.max(), rtol=1e-9)
    np.testing.assert_allclose(info["mean_abs_error"], e.mean(), rtol=1e-9)
    np.testing.assert_allclose(info["mean_squared_error"], np.mean(e**2), rtol=1e-9)
    np.testing.assert_allclose(h["poles"][-1], poles, rtol=1e-12)
    np.testing.assert_allclose(h["residues"][-1], residues, rtol=1e-9, atol=1e-12 * np.abs(residues).max())
    assert h["mean_squared_error"][-1] == info["mean_squared_error"]
    # every iterate's errors are those of its recorded model
    k = n // 2
    ek = np.abs(np.asarray(data) - np.asarray(response(h["poles"][k], h["residues"][k], h["feedthrough"][k])))
    np.testing.assert_allclose(h["max_abs_error"][k], ek.max(), rtol=1e-9)
    assert h["max_abs_error"][-1] < h["max_abs_error"][0]


def test_discrete_fit_default_output_and_info():
    data = discrete_data()
    model = vector_fitting_discrete(6, data, F, F0, FS, use_cache=False)
    assert len(model) == 3
    *with_info, info = vector_fitting_discrete(6, data, F, F0, FS, return_info=True, use_cache=False)
    for a, b in zip(model, with_info):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    assert info["iterations"] == 10 and not info["converged"]
    check_info(info, model, data, z_response)


def test_continuous_fit_default_output_and_info():
    data = continuous_data()
    model = vector_fitting_continuous(6, data, F, F0, use_cache=False)
    assert len(model) == 3
    *with_info, info = vector_fitting_continuous(6, data, F, F0, return_info=True, use_cache=False)
    for a, b in zip(model, with_info):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    check_info(info, model, data, s_response)


def test_early_stop_trims_the_history():
    data = discrete_data()
    *model, info = vector_fitting_discrete(6, data, F, F0, FS, weight_threshold=1e3, return_info=True, use_cache=False)
    assert info["iterations"] == 1 and info["converged"]
    assert info["history"]["poles"].shape[0] == 2
    np.testing.assert_allclose(info["history"]["poles"][-1], model[0], rtol=1e-12)


@pytest.mark.parametrize("domain", ["z", "s"])
def test_order_search_info(domain):
    if domain == "z":
        data, response = discrete_data(), z_response
        search = lambda **kw: optimize_order_vector_fitting_discrete(3, 12, data, F, F0, FS, use_cache=False, **kw)
    else:
        data, response = continuous_data(), s_response
        search = lambda **kw: optimize_order_vector_fitting_continuous(3, 12, data, F, F0, use_cache=False, **kw)
    model = search()
    assert len(model) == 3
    *with_info, info = search(return_info=True)
    for a, b in zip(model, with_info):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    o = info["order_search"]
    assert (o["min_order"], o["max_order"]) == (3, 12)
    assert o["selected_order"] == model[0].shape[0]
    orders = [n for n, _ in o["evaluated"]]
    assert orders[:3] == [3, 12, 11] and len(set(orders)) == len(orders)
    assert o["complexity_penalty"] > 0
    check_info(info, model, data, response)
