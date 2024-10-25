"""Ideal time-domain models."""

import jax.numpy as jnp
import sax
from jax.typing import ArrayLike

def coupler(
    coupling: float = 0.5,
    loss: float = 0.0,
    phi: float = jnp.pi / 2,
):
    pass

def waveguide(
    wl: ArrayLike | float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0,
):
    pass