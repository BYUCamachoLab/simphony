"""Simulation module."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
from jax.typing import ArrayLike
from sax.saxtypes import Model

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simphony.circuit import Circuit


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


class SParameterSimulation:
    def __init__(self, ckt: Circuit, settings: dict = None):
        if settings is not None:
            self.update_settings(settings)

        self.circuit = ckt
        self._determine_dc_voltage_order()

        self.update_settings(settings)

    def run(self, settings: dict = None):
        if settings is not None:
            self.update_settings(settings)

        self._calculate_dc_voltages()
        self._calculate_scattering_matrices()

    def update_settings(self):
        """
        Useful when running parameter sweeps
        """
        pass

    def _determine_dc_voltage_order(self):
        """
        Voltage signals at electrical ports are assumed to be constant
        for SParameterSimulations, but they are not known a priori, unless
        the voltage source is not dependent on an input signal.

        Since electrical connections are uni-directional, this function is
        able to find the order in which electrical component voltages must
        be calculated to find the proper steady state.
        """
        electrooptic_components = []
        graph = self.circuit.graph
        models = self.circuit.models
        for node, attr in graph.nodes(data=True):
            model = attr["component"]
            component = models[model]
            if not inspect.isclass(component):
                break

            if component.electrical_ports and component.optical_ports:
                electrooptic_components.append(graph.nodes(node))

        pass

    def _calculate_dc_voltages(self):
        pass

    def _calculate_scattering_matrices(self):
        pass
