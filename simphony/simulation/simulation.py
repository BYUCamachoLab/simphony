"""Simulation module."""

from __future__ import annotations

# import inspect

import jax.numpy as jnp
from jax.typing import ArrayLike
from sax.saxtypes import Model
from typing import Annotated

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simphony.circuit import Circuit

from copy import deepcopy
import jax 

from flax import struct
from dataclasses import field

class SimDevice:
    """Base class for all source or measure devices."""

    # TODO: Add bandwidth option to classical
    def __init__(self, ports: list) -> None:
        self.ports = ports

@struct.dataclass
class SimulationParameters:
    # def __init__(
    #     self,
    sampling_period:float=1e-15
    # sampling_rate:float=1e15,
    num_time_steps:int =int(1e4)
    prng_key: Annotated[jax.Array, "shape=(2,), dtype=jax.uint32"]=field(default_factory=lambda: jax.random.PRNGKey(0))
    num_optical_modes:int = int(1) 
    # prng_key: Annotated[jax.Array, "shape=(2,), dtype=jax.uint32"]=jax.random.key(0)
    # ):
    #     super().__setattr__('_locked', False)
    #     self.sampling_period = sampling_period
    #     self.sampling_rate = sampling_rate
    #     self.num_time_steps = num_time_steps
    #     self.prng_key=prng_key
    #     if prng_key is None:
    #         self.prng_key = jax.random.key(0)
    #     super().__setattr__('_locked', True)

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
    
    def _instantiate_components(self, settings):
        self.components = {}
        for component_name in self.circuit.graph.nodes:
            model_name = self.circuit.netlist['instances'][component_name]['component']
            model = self.circuit.models[model_name]
            component_settings = settings[component_name]
            self.components[component_name] = model(**component_settings)
    
    def _clear_settings(self):
        self.settings = {}
        for instance in self.circuit.graph.nodes:
                self.settings[instance] = {}

    def reset_settings(self, use_default_settings: bool = True):
        """
        Reset settings to their defaults (specified in Circuit) or clear all settings
        """
        if use_default_settings:
            self._clear_settings()
            additional_settings = deepcopy(self.circuit.default_settings)
            self.add_settings(additional_settings)
        else:
            self._clear_settings()

    def add_settings(self, settings: dict):
        """
        Update the current settings with additional settings.
        """
        for instance, instance_settings in settings.items():
            self.settings[instance].update(instance_settings)


class SimulationResult:
    """Base class for simphony simulation results."""