"""Tests for a-priori model-order bounds (simphony.time_domain.vector_fitting.order_estimation)."""

import warnings

import numpy as np
import pytest
from scipy.constants import speed_of_light

from simphony.conventions import PHYSICIST
from simphony.libraries.ideal.s_parameters import merge_vector_fitting_parameters, optical_s_parameter
from simphony.libraries.siepic import directional_coupler
from simphony.simulation.block_mode import BlockModeSimulationParameters
from simphony.time_domain.vector_fitting.order_estimation import (
    delay_bandwidth_cycles,
    estimate_order_bounds,
    loewner_singular_values,
)
from simphony.time_domain.vector_fitting.z_domain import pole_residue_response_discrete

F0 = speed_of_light / 1.55e-6
FS = 1e14


def _rational_data(order, q=2, m=2, seed=0, n=1000):
    rng = np.random.default_rng(seed)
    poles = rng.uniform(0.5, 0.95, order) * np.exp(1j * rng.uniform(-np.pi, np.pi, order))
    residues = 0.05 * (rng.normal(size=(order, q, m)) + 1j * rng.normal(size=(order, q, m)))
    feedthrough = 0.3 * (rng.normal(size=(q, m)) + 1j * rng.normal(size=(q, m)))
    f = F0 + np.linspace(-0.4, 0.4, n) * FS
    H = pole_residue_response_discrete(f, F0, FS, poles, residues, feedthrough, sign_convention=PHYSICIST)
    return np.asarray(H), f


def test_loewner_rank_recovers_the_mcmillan_degree():
    # A rational model of order 12 (2x2, generic residues): exactly 12 singular values above the rounding floor.
    H, f = _rational_data(12)
    s = loewner_singular_values(H, f, F0, FS, PHYSICIST)
    assert np.count_nonzero(s > 1e-9) == 12
    est = estimate_order_bounds(H, f, F0, FS, PHYSICIST, min_tolerance=1e-9, max_tolerance=1e-11)
    assert est.min_order == 12 and est.max_order >= est.min_order + 2


def test_delay_bandwidth_counts_fringes():
    # Two paths 20 ps apart over 2 THz: 40 phase cycles of the difference.
    f = F0 + np.linspace(-1e12, 1e12, 2001)
    H = (0.5 * np.exp(2j * np.pi * (f - F0) * 5e-12) + 0.5 * np.exp(2j * np.pi * (f - F0) * 25e-12))[:, None, None]
    assert delay_bandwidth_cycles(H, f) == pytest.approx(40, rel=0.1)


def _dc(**vf):
    params = BlockModeSimulationParameters(dt=1 / FS, num_time_steps=8, mode_identifiers=("te",))
    element = optical_s_parameter(directional_coupler)(params, sax_settings={}, vector_fitting_parameters=vf)
    return element._state_space(params)


def test_unset_bounds_use_the_estimate_and_low_explicit_min_warns():
    f = np.linspace(speed_of_light / 1.6e-6, speed_of_light / 1.5e-6, 1000)
    sd = directional_coupler(wl=1e6 * speed_of_light / f)
    ports = ["port_1", "port_2", "port_3", "port_4"]
    S = np.stack([np.stack([np.asarray(sd[(o, i)]) for i in ports], -1) for o in ports], 1)
    est = estimate_order_bounds(S, f, F0, FS, PHYSICIST)
    assert 2 < est.min_order < est.max_order
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _dc()                                                  # estimated bounds: no warning
        _dc(min_model_order=est.min_order, max_model_order=est.min_order + 6)
    with pytest.warns(UserWarning, match="below the a-priori estimate"):
        _dc(min_model_order=2, max_model_order=est.min_order + 6)


def test_fused_groups_estimate_when_any_member_leaves_a_bound_unset():
    params = BlockModeSimulationParameters(dt=1 / FS, num_time_steps=8, mode_identifiers=("te",))
    cls = optical_s_parameter(directional_coupler)
    members = {"a": cls(params, sax_settings={}, vector_fitting_parameters={}),
               "b": cls(params, sax_settings={}, vector_fitting_parameters={"model_order": 20})}
    merged = merge_vector_fitting_parameters("g", members)
    assert merged["model_order"] is None
    assert merged["min_model_order"] is None and merged["max_model_order"] is None
