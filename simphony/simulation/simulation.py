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
        self.wl = wl

    def run(self):
        """Run the simulation."""
        raise NotImplementedError


class SimulationResult:
    pass


# class ClassicalSim(Simulation):
#     """
#     Classical simulation
#     """

#     def __init__(self,  ckt: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(ckt, wl)


class MonteCarloSim(Simulation):
    """Monte Carlo simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        super().__init__(ckt, wl)


class LayoutAwareSim(Simulation):
    """Layout-aware simulation."""

    def __init__(self, cir: Circuit, wl: jnp.ndarray) -> None:
        super().__init__(cir, wl)


class SamplingSim(Simulation):
    """Sampling simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        super().__init__(ckt, wl)


class TimeDomainSim(Simulation):
    """Time-domain simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        super().__init__(ckt, wl)


class QuantumSim(Simulation):
    """Quantum simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        super().__init__(ckt, wl)
