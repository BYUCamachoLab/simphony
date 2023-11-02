"""Simulaion module."""

import jax.numpy as jnp
from jax.typing import ArrayLike
from sax.saxtypes import Model


class Simulation:
    """Base class for simphony simulations."""

    def __init__(self, ckt: Model, wl: ArrayLike) -> None:
        self.ckt = ckt
        self.wl = jnp.asarray(wl).reshape(-1)

    def run(self):
        """Run the simulation."""
        raise NotImplementedError


class SimulationResult:
    pass
