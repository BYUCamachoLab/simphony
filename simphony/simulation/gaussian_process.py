"""Gaussian process simulation for photonic circuits.

Mirrors BlockModeSimulation but propagates GaussianProcessOpticalSignal
objects (mean + covariance) instead of BlockModeOpticalSignal objects
(mean only).
"""

from __future__ import annotations

from dataclasses import field

import jax
import jax.numpy as jnp
import networkx as nx
from flax import struct

from simphony.circuit.circuit import Circuit
from simphony.simulation.simulation import (
    Simulation,
    SimulationMode,
    SimulationParameters,
    SimulationResult,
)


@struct.dataclass
class GaussianProcessSimulationParameters(SimulationParameters):
    """Parameters for GaussianProcessSimulation.

    Attributes
    ----------
    dt : float
        Sampling period in seconds.
    num_time_steps : int
        Number of discrete time steps T to simulate.
    num_ir_taps : int
        Number of impulse response taps K used when computing h from the
        state-space.  Increasing K improves accuracy for long-memory systems.
    optical_baseband_wavelengths : jax.Array
        Carrier wavelengths in metres for which to track signal statistics.
    """

    simulation_mode: SimulationMode = field(
        default_factory=lambda: SimulationMode.GAUSSIAN_PROCESS
    )
    directed: bool = True
    dt: float = 1e-14
    num_time_steps: int = 1000
    num_ir_taps: int = 200
    optical_baseband_wavelengths: jax.Array = field(
        default_factory=lambda: jnp.array([1.55e-6])
    )


class GaussianProcessSimulationResult(SimulationResult):
    """Stores the tracked-port signals from a GaussianProcessSimulation run."""

    def __init__(self, input_signals: dict, output_signals: dict):
        self.input_signals = input_signals
        self.output_signals = output_signals


class GaussianProcessSimulation(Simulation):
    """Directed Gaussian process simulation of a photonic circuit.

    Propagates both the mean field and the temporal covariance of each optical
    signal through the circuit using the Papoulis equations
    (see `simphony.time_domain.stochastic.gaussian_process`).

    The execution model mirrors BlockModeSimulation exactly:
    1. Instantiate and topologically sort the directed circuit graph.
    2. For each component in topological order, call
       ``component._gaussian_process_mode_response(inputs, params)``.
    3. Collect tracked-port signals into a GaussianProcessSimulationResult.

    Parameters
    ----------
    circuit : Circuit
        The photonic circuit to simulate.
    settings : dict
        Per-instance settings (sax_settings, vector_fitting_parameters, etc.).
    tracked_ports : dict, optional
        Mapping of user-facing port names to ``"instance,port"`` designators.
        If None, all circuit ports are tracked.
    simulation_parameters : GaussianProcessSimulationParameters, optional
        Simulation configuration.  Defaults are used if not provided.
    """

    def __init__(
        self,
        circuit: Circuit,
        settings: dict,
        tracked_ports: dict = None,
        simulation_parameters: GaussianProcessSimulationParameters = None,
    ):
        if settings is None:
            settings = {}
        if simulation_parameters is None:
            simulation_parameters = GaussianProcessSimulationParameters()

        self.simulation_parameters = simulation_parameters
        self.circuit = circuit
        self.settings = settings
        self.tracked_ports = tracked_ports
        self.component_inputs: dict = {}
        self.component_outputs: dict = {}

    def run(self) -> GaussianProcessSimulationResult:
        """Run the simulation and return a GaussianProcessSimulationResult."""
        self._instantiated_circuit = self.circuit.instantiate(
            self.settings,
            self.simulation_parameters,
            tracked_ports=self.tracked_ports,
            directed=True,
        )

        order = self._determine_gaussian_process_order(self._instantiated_circuit)

        for instance_name in order:
            self._collect_component_inputs(instance_name)
            inputs = self.component_inputs[instance_name]
            component = self._instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ][instance_name]["model"]
            outputs = component._gaussian_process_mode_response(
                inputs, self.simulation_parameters
            )
            self.component_outputs[instance_name] = outputs

        input_signals: dict = {}
        output_signals: dict = {}
        for (
            tracked_name,
            designator,
        ) in self._instantiated_circuit.port_lookup_table.items():
            instance_name, port_name = designator.split(",")
            if port_name in self.component_inputs.get(instance_name, {}):
                input_signals[tracked_name] = self.component_inputs[instance_name][
                    port_name
                ]
            if port_name in self.component_outputs.get(instance_name, {}):
                output_signals[tracked_name] = self.component_outputs[instance_name][
                    port_name
                ]

        return GaussianProcessSimulationResult(input_signals, output_signals)

    def _collect_component_inputs(self, component: str) -> None:
        inputs = {}
        for upstream in [
            u for u, _ in self._instantiated_circuit.graph.in_edges(component)
        ]:
            edge_data = self._instantiated_circuit.graph.get_edge_data(
                upstream, component
            )
            for _, edge in edge_data.items():
                inputs[edge["dst_port"]] = self.component_outputs[upstream][
                    edge["src_port"]
                ]
        self.component_inputs[component] = inputs

    def _determine_gaussian_process_order(self, instantiated_circuit) -> list:
        """Topological sort — same logic as BlockModeSimulation."""
        try:
            return list(nx.topological_sort(instantiated_circuit.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError(
                "Failed to determine simulation order — circular dependencies detected"
            )
