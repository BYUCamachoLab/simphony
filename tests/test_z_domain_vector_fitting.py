"""z-domain state-space frequency response honours the sign convention."""

import jax.numpy as jnp
import numpy as np
import pytest

from simphony.conventions import ENGINEER, PHYSICIST
from simphony.time_domain.vector_fitting.z_domain import (
    pole_residue_response_discrete,
    state_space_discrete,
    state_space_frequency_response_discrete,
)

F0 = 193.4e12
FS = 1e12


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_state_space_frequency_response_matches_pole_residue_response(sign_convention):
    rng = np.random.default_rng(0)
    poles = jnp.asarray(rng.uniform(0.5, 0.95, 5) * np.exp(1j * rng.uniform(-3, 3, 5)))
    residues = jnp.asarray(rng.normal(size=(5, 2, 3)) + 1j * rng.normal(size=(5, 2, 3)))
    feedthrough = jnp.asarray(rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
    f = jnp.asarray(F0 + np.linspace(-0.45, 0.45, 41) * FS)

    A, B, C, D = state_space_discrete(poles, residues, feedthrough)

    np.testing.assert_allclose(
        state_space_frequency_response_discrete(A, B, C, D, f, F0, 1 / FS, sign_convention=sign_convention),
        pole_residue_response_discrete(f, F0, FS, poles, residues, feedthrough, sign_convention=sign_convention),
        rtol=1e-10,
    )


def random_continuous_system(seed=0, n=5, m=3, q=2, real=False):
    """A small stable continuous state-space model (dense A, shifted into the left half-plane), A in rad/s."""
    rng = np.random.default_rng(seed)

    def draw(*shape):
        x = rng.normal(size=shape)
        return x if real else x + 1j * rng.normal(size=shape)

    A = draw(n, n) * 2e11
    A = A - (np.max(np.linalg.eigvals(A).real) + 1e11) * np.eye(n)
    return A, draw(n, m) * 1e11, draw(q, n), draw(q, m)


@pytest.mark.parametrize("method", ["bilinear", "zoh", "foh"])
def test_continuous_to_discrete_matches_scipy(method):
    from scipy.signal import cont2discrete

    from simphony.time_domain.vector_fitting.z_domain import continuous_to_discrete_state_space

    A, B, C, D = random_continuous_system()
    expected = cont2discrete((A, B, C, D), 1 / FS, method=method)[:4]

    for got, want in zip(continuous_to_discrete_state_space(A, B, C, D, FS, method=method), expected):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12 * np.max(np.abs(want)))


@pytest.mark.parametrize("sign_convention", [PHYSICIST, ENGINEER])
def test_bilinear_state_space_matches_pole_residue_bilinear_transform(sign_convention):
    from simphony.time_domain.vector_fitting.s_domain import state_space_continuous
    from simphony.time_domain.vector_fitting.z_domain import (
        bilinear_transform,
        continuous_to_discrete_state_space,
    )

    rng = np.random.default_rng(1)
    poles = jnp.asarray(sign_convention * 1j * 2 * np.pi * rng.uniform(-0.3, 0.3, 6) * FS - rng.uniform(0.05, 0.3, 6) * FS)
    residues = jnp.asarray((rng.normal(size=(6, 2, 3)) + 1j * rng.normal(size=(6, 2, 3))) * FS)
    feedthrough = jnp.asarray(rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)))
    f = jnp.asarray(F0 + np.linspace(-0.45, 0.45, 41) * FS)

    Ad, Bd, Cd, Dd = continuous_to_discrete_state_space(
        *state_space_continuous(poles, residues, feedthrough), FS, method="bilinear"
    )
    discrete_model = bilinear_transform(poles, residues, feedthrough, FS)

    np.testing.assert_allclose(
        state_space_frequency_response_discrete(Ad, Bd, Cd, Dd, f, F0, 1 / FS, sign_convention=sign_convention),
        pole_residue_response_discrete(f, F0, FS, *discrete_model, sign_convention=sign_convention),
        rtol=1e-9,
    )
