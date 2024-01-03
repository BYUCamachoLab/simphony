"""Simulation module."""

import jax.numpy as jnp
from jax.typing import ArrayLike
from sax.saxtypes import Model


class SimDevice:
    """Base class for all source or measure devices."""

    # TODO: Add bandwidth option to classical
    def __init__(self, ports: list) -> None:
        self.ports = ports


class Simulation:
    """Base class for simphony simulations.

    Parameters
    ----------
    ckt : Model
        A callable SAX model.
    wl : ArrayLike
        The wavelengths at which to simulate the circuit.
    """

    def __init__(self, ckt: Model, wl: ArrayLike) -> None:
        self.ckt = ckt
        self.wl = jnp.asarray(wl).reshape(-1)

    def run(self):
        """Run the simulation."""
        raise NotImplementedError


class SimulationResult:
    """Base class for simphony simulation results."""
