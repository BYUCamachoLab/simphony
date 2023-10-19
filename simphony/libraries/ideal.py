import jax.numpy as jnp
import sax
from jax.typing import Array, ArrayLike


def coupler(
    *,
    coupling: float = 0.5,
) -> sax.SDict:
    """A simple ideal coupler model.

    Ports are arranged as follows:

        o2 ---\        /--- o3
               ========
        o0 ---/        \--- o1
    """
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("o2", "o3"): tau,
            ("o2", "o1"): 1j * kappa,
            ("o0", "o3"): 1j * kappa,
            ("o0", "o1"): tau,
        }
    )
    return sdict


def straight(
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

    Args:
        wl: Wavelength in microns.
        wl0: Center wavelength in microns.
        neff: Effective index.
        ng: Group index.
        length: Length in microns.
        loss: Loss in dB/cm.

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
