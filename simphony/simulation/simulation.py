"""Simulaion module."""

from simphony.circuit import Circuit
from simphony.models import Model

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    from simphony.utils import jax

    JAX_AVAILABLE = False


class Simulation:
    """Base class for simphony simulations."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        self.ckt = ckt
        self.wl = jnp.asarray(wl).reshape(-1)

    def run(self):
        """Run the simulation."""
        raise NotImplementedError


class SimulationResult:
    pass
