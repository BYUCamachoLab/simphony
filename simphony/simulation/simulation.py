"""Simulation module."""

from __future__ import annotations

# import inspect

import jax.numpy as jnp
from jax.typing import ArrayLike
from sax.saxtypes import Model


# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simphony.circuit import Circuit

from copy import deepcopy
import jax 

from flax import struct
from dataclasses import field

from sax import DEFAULT_MODES

from enum import StrEnum

class SimulationMode(StrEnum):
    """Names for supported simulator execution modes.

    PCells and component factories use this enum to choose simulator-specific
    internal designs without importing individual simulator classes.
    """
    S_PARAMETER = "s_parameter"
    SAMPLE_MODE = "sample_mode"
    BLOCK_MODE = "block_mode"
    STEADY_STATE = "steady_state"
    GAUSSIAN_PROCESS = "gaussian_process"
    _SIMPHONY_PREPROCESSING = "simphony_preprocessing"
    #### TODO: Fix naming conventions for all simulation modes
    # TRANSIENT_SAMPLE = "transient_sample"
    # TRANSIENT_BLOCK = "transient_block"

class SimDevice:
    """Base class for all source or measure devices."""

    # TODO: Add bandwidth option to classical
    def __init__(self, ports: list) -> None:
        self.ports = ports

@struct.dataclass
class SimulationParameters:
    """Base dataclass for parameters shared by simulator variants.

    Subclasses add mode-specific fields such as wavelength grids, time step, or
    number of time samples.

    Attributes
    ----------
    simulation_mode:
        `SimulationMode` value identifying the active simulator.
    directed:
        Whether the simulation requires a directed instance graph.
    mode_identifiers:
        Optical mode labels represented by optical signal arrays.
    seed:
        Integer seed used by simulations/components that create PRNG keys.
    """
    # def __init__(
    #     self,
    simulation_mode: SimulationMode = None
    directed: bool = None
    # sampling_period:float=1e-15
    # sampling_rate:float=1e15,
    # num_time_steps:int =int(1e4)
    # prng_key: Annotated[jax.Array, "shape=(2,), dtype=jax.uint32"]=field(default_factory=lambda: jax.random.PRNGKey(0))
    mode_identifiers: list = field(default_factory=lambda: DEFAULT_MODES)
    seed = 0
    
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
    """Base class for Simphony simulation drivers."""

    def __init__(self, ckt: Model, wl: ArrayLike) -> None:
        self.ckt = ckt
        self.wl = jnp.asarray(wl).reshape(-1)

    def run(self):
        """Run the simulation."""
        raise NotImplementedError
    
    def _instantiate_components(self, settings):
        self.components = {}
        for instance_name in self.circuit.graph.nodes:
            model_name = self.circuit.netlist['instances'][instance_name]['component']
            model = self.circuit.models[model_name]
            component_settings = settings[instance_name]
            self.components[instance_name] = model(**component_settings)
    
    def _clear_settings(self):
        self.settings = {}
        for instance in self.circuit.graph.nodes:
                self.settings[instance] = {}

    def reset_settings(self, use_default_settings: bool = True):
        """Reset per-instance settings.

        Parameters
        ----------
        use_default_settings:
            If true, restore defaults from the circuit before applying future
            updates. If false, clear every instance settings dictionary.
        """
        if use_default_settings:
            self._clear_settings()
            additional_settings = deepcopy(self.circuit.default_settings)
            self.add_settings(additional_settings)
        else:
            self._clear_settings()

    def add_settings(self, settings: dict):
        """Merge additional per-instance settings into the simulation.

        `settings` is keyed by instance name. Values are shallow-merged into the
        current settings for each instance.
        """
        for instance, instance_settings in settings.items():
            self.settings[instance].update(instance_settings)


class SimulationResult:
    """Base class for simphony simulation results."""
