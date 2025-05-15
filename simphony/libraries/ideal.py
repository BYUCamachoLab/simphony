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
            ("o0", "o3"): kappa,
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
    # amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    loss_mag = loss / (10 * jnp.log10(jnp.exp(1)))
    alpha = loss_mag * 1e-4
    amplitude = jnp.asarray(jnp.exp(-alpha * length / 2), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("o0", "o1"): transmission,
        }
    )
    return sdict


def phase_modulator(
    *,
    mod_signal: ArrayLike|float = 0.0,
    k_p: float = 1.0,
)-> sax.SDict:
    
    """
    Parameters:
    - freq: Frequency array (input carrier frequencies).
    - mod_signal: Modulating signal array (same length as freq).
    - kp: Phase modulation constant (controls sensitivity).

    Returns:
    - s_dict: Scattering matrix dictionary for the phase modulator.
    """
    # Calculate the phase shift
    phase_shift = k_p * mod_signal  # Instantaneous phase shift

    # Scattering parameter from input to output (phase modulation applied)
    s_input_output = jnp.exp(1j * phase_shift)

    # Define the s_dict structure
    s_dict = {
        ("o0", "o1"): s_input_output,
        ("o1","o0"): s_input_output,  # Transmission from input to output
    }

    return s_dict



def MultiModeInterferometer(
    *,
    wl: float | jnp.ndarray = 1.55,
    length: float = 10.0,
    loss: float = 0.0,
    r: int = 2,
    s: int = 2,
) -> sax.SDict:
    """
    Return the S-dictionary for an r×s Multimode Interference coupler,
    at reduced self-imaging length, with optional uniform loss.
    """

    # total ports
    N_size = r + s

    # 1) Build the phase matrix (quadratic law)
    phases = jnp.zeros((N_size, N_size))
    for i in range(1, r+1):
        for j in range(1, s+1):
            if (i+j) % 2 == 0:
                phi = -(jnp.pi/(4*r)) * (j - i) * (2*r + i - j)
            else:
                phi = -(jnp.pi/(4*r)) * (i + j - 1) * (2*r - i - j + 1)
            # fill both symmetric entries
            phases = phases.at[i-1, N_size-j].set(phi)
            phases = phases.at[N_size-j, i-1].set(phi)

    # 2) Compute amplitude attenuation (same for all couplings)
    loss_mag = loss / (10 * jnp.log10(jnp.exp(1)))
    alpha    = loss_mag * 1e-4
    amp      = jnp.exp(-alpha * length / 2)  # scalar real
    ones     = jnp.ones_like(wl, dtype=complex)

    # 3) Build the forward S-dictionary (only inputs 0…r-1 → outputs r…r+s-1)
    s_dict = {}
    for inp in range(r):
        for out in range(r, r+s):
            φ    = phases[inp, out]
            gain = amp / jnp.sqrt(s) * jnp.exp(1j * φ)
            s_dict[(f"o{inp}", f"o{out}")] = gain * ones

    # 4) Mirror it back to get a fully reciprocal device
    return sax.reciprocal(s_dict)

def make_mmi_model(*, r: int, s: int,
                   default_wl: float = 1.55,
                   default_length: float = 10.0,
                   default_loss: float   = 0.0):
    """
    Factory that returns an MMI_model(recipient of no-args or wl/length/loss).
    """

    def MMI_model(
        *,
        wl:     ArrayLike | float = default_wl,
        length: float              = default_length,
        loss:   float              = default_loss,
    ) -> sax.SDict:
        return MultiModeInterferometer(
            wl=wl,
            length=length,
            loss=loss,
            r=r,
            s=s
        )

    # give it a meaningful name
    MMI_model.__name__ = f"MMI_{r}x{s}"
    return MMI_model

