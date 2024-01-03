# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Ideal circuit models."""

import jax.numpy as jnp
import sax
from jax.typing import ArrayLike


def coupler(
    *,
    coupling: float = 0.5,
    loss: float = 0.0,
    phi: float = jnp.pi / 2,
) -> sax.SDict:
    """A simple ideal coupler model.

    Ports are arranged as follows::

        o2 ---\        /--- o3
               --------
               --------
        o0 ---/        \--- o1

    Parameters
    ----------
    coupling : float
        Power coupling coefficient (default 0.5).
    loss : float
        Loss in dB (default 0.0). Positive values indicate loss.
    phi : float
        Phase shift of the reflected signal (default jnp.pi/2).

    Returns
    -------
    sdict : sax.SDict
        A dictionary of scattering matrices.
    """

    kappa = coupling**0.5 * 10 ** (-loss / 20) * jnp.exp(1j * phi)
    tau = (1 - coupling) ** 0.5 * 10 ** (-loss / 20)
    sdict = sax.reciprocal(
        {
            ("o2", "o3"): tau,
            ("o2", "o1"): kappa,
            ("o0", "o3"): jnp.conj(kappa),
            ("o0", "o1"): tau,
        }
    )
    return sdict


def waveguide(
    *,
    wl: ArrayLike | float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0,
) -> sax.SDict:
    """A simple straight waveguide model.

    Port names are "o0" and "o1".

    Parameters
    ----------
    wl : ArrayLike or float
        Wavelength in microns (default 1.55).
    wl0 : float
        Center wavelength in microns (default 1.55).
    neff : float
        Effective index (default 2.34).
    ng : float
        Group index (default 3.4).
    length : float
        Length in microns (default 10.0).
    loss : float
        Loss in dB/cm (default 0.0).

    Returns:
        A dictionary of scattering matrices.
    """
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("o0", "o1"): transmission,
        }
    )
    return sdict
