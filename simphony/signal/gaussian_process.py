"""Gaussian process signal types for GaussianProcessSimulation."""

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class GaussianProcessOpticalSignal:
    """Optical signal for Gaussian process simulation.

    Carries both the deterministic mean field and the stochastic covariance
    at every time step, enabling full second-order Gaussian state tracking.

    Attributes
    ----------
    mean_amplitude : jnp.ndarray, shape (T, L, M)
        Mean complex field amplitude.
        T = num_time_steps, L = num wavelengths, M = num polarisation modes.
    covariance : jnp.ndarray, shape (L, T, T, M, M)
        Per-wavelength temporal covariance.
        covariance[l, n1, n2, i, j] = E[(x[n1,i]-µ[n1,i]) conj(x[n2,j]-µ[n2,j])]
        at wavelength index l for polarisation modes i, j.
        Different wavelengths are treated as statistically independent.
    wavelength : jnp.ndarray, shape (L,)
        Carrier wavelengths in metres.
    """

    mean_amplitude: jnp.ndarray  # (T, L, M)
    covariance: jnp.ndarray  # (L, T, T, M, M)
    wavelength: jnp.ndarray  # (L,)
