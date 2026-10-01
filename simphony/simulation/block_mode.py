import logging
from copy import deepcopy
from dataclasses import field
from time import time

import jax
import jax.numpy as jnp
import networkx as nx
from flax import struct

from simphony.circuit.circuit import Circuit
from simphony.circuit.rates import infer_sample_rates, local_parameters
from simphony.simulation.simulation import (
    Simulation,
    SimulationMode,
    SimulationParameters,
    SimulationResult,
)

logger = logging.getLogger(__name__)


@struct.dataclass
class BlockModeSimulationParameters(SimulationParameters):
    """Global settings for a Block mode simulation.

    Block mode evaluates a full time block at once. These parameters define the
    time grid, optical carrier wavelengths, and optical modes shared by every
    component in the circuit.

    Attributes
    ----------
    dt:
        Time step, in seconds, between adjacent samples in the block.
    num_time_steps:
        Number of samples processed by each component during one simulation run.
    optical_baseband_wavelengths:
        Carrier wavelengths, in meters, tracked by the optical envelope arrays.
    mode_identifiers:
        Inherited from `SimulationParameters`; labels for the optical modes
        represented on the last axis of a `BlockModeOpticalSignal`.
    use_state_space_optimization:
        Enables optimized structured state-space updates where available.
    time_offset:
        Time, in seconds, of the first sample. Multirate simulations give each
        rate region its own copy of the parameters, with that region's `dt`,
        `num_time_steps` and `time_offset` (see `simphony.circuit.rates`).
    backward_pass:
        If true, run a second, backward pass after the forward pass. The
        backward-travelling waves emitted on input ports during the forward
        pass (reflections) are propagated back through the circuit in reverse
        order, accumulating the reflections of every earlier stage. The
        result is first order in back-reflection: waves reflected back into
        the forward direction during the backward pass are discarded. S-parameter
        elements fit their full S-matrix when this is enabled, and only
        their forward `S[output <- input]` block otherwise.
    """

    simulation_mode: SimulationMode = field(
        default_factory=lambda: SimulationMode.BLOCK_MODE
    )
    directed: bool = True
    dt: float = 1e-14
    num_time_steps: int = 1000
    optical_baseband_wavelengths: jax.Array = field(
        default_factory=lambda: jax.numpy.array([1.55e-6])
    )
    use_state_space_optimization: bool = True
    time_offset: float = 0.0
    backward_pass: bool = False


class BlockModeSimulationResult(SimulationResult):
    """Signals collected from a completed Block mode simulation.

    Attributes
    ----------
    input_signals:
        Signals observed on tracked ports when the tracked port corresponds to a
        component input.
    output_signals:
        Signals observed on tracked ports when the tracked port corresponds to a
        component output.
    backward_signals:
        Backward-travelling waves observed on tracked ports. Only populated
        when `BlockModeSimulationParameters.backward_pass` is true. On a tracked
        input port this is the total wave travelling back out of the circuit
        (the reflection seen by whatever drives that port).
    """

    def __init__(self, input_signals, output_signals, backward_signals=None):
        self.input_signals = input_signals
        self.output_signals = output_signals
        self.backward_signals = backward_signals if backward_signals else {}


class BlockModeSimulation(Simulation):
    """Run a directed circuit by propagating full-block signals component by
    component.

    The simulator instantiates the circuit with the provided settings, determines
    a topological execution order, calls each component's block-mode response, and
    returns the signals available at tracked or top-level ports. When
    `simulation_parameters.backward_pass` is true, a second pass propagates
    backward-travelling waves in reverse order (see
    `BlockModeSimulationParameters`).

    Parameters
    ----------
    circuit:
        Circuit to simulate. For Block mode, the instantiated graph must be
        directed and acyclic.
    settings:
        Per-instance settings used when the circuit is instantiated. SAX
        S-parameter components typically need `sax_settings`,
        `port_directionality`, and `vector_fitting_parameters`.
    tracked_ports:
        Optional mapping of names to internal `"instance,port"` designators to
        expose in the returned result. Top-level circuit ports are tracked by
        default.
    simulation_parameters:
        Shared `BlockModeSimulationParameters`. If omitted, defaults are used.
    fuse_s_parameters:
        Fuse adjacent SAX S-parameter elements (sharing an
        `s_parameter_group` setting) into single elements before simulating.
        This reduces the number of vector fits.
    s_parameter_group_settings:
        Optional mapping from `s_parameter_group` id to settings (for example
        `vector_fitting_parameters`) for the fused elements of that group.
    """

    def __init__(
        self,
        circuit: Circuit,
        settings,
        tracked_ports: dict = None,
        simulation_parameters=None,
        fuse_s_parameters: bool = False,
        s_parameter_group_settings: dict = None,
    ):
        if settings is None:
            settings = {}
        if simulation_parameters is None:
            simulation_parameters = BlockModeSimulationParameters()

        self.simulation_parameters = simulation_parameters
        self.circuit = deepcopy(circuit)
        self.settings = deepcopy(settings)
        self.tracked_ports = deepcopy(tracked_ports)
        self.fuse_s_parameters = fuse_s_parameters
        self.s_parameter_group_settings = deepcopy(s_parameter_group_settings)
        self.component_inputs = {}
        self.component_outputs = {}
        self.forward_reflections = {}
        self.backward_inputs = {}
        self.backward_outputs = {}

    def run(
        self,
    ) -> BlockModeSimulationResult:
        """Run the Block mode simulation and collect tracked port signals.

        The circuit is instantiated in directed mode, components are
        evaluated in topological order, and each component receives a
        full time block of input signals at once.
        """
        self.component_inputs = {}
        self.component_outputs = {}
        self.forward_reflections = {}
        self.backward_inputs = {}
        self.backward_outputs = {}

        tic = time()
        self._instantiated_circuit = self.circuit.instantiate(
            self.settings,
            self.simulation_parameters,
            tracked_ports=self.tracked_ports,
            fuse_s_parameters=self.fuse_s_parameters,
            s_parameter_group_settings=self.s_parameter_group_settings,
        )
        self.block_mode_order = self._determine_block_mode_order_nx_method(
            self._instantiated_circuit
        )
        self._setup_rates()
        logger.debug("Block mode execution order: %s", self.block_mode_order)

        for instance_name in self.block_mode_order:
            self._collect_component_inputs(instance_name)
            inputs = self.component_inputs[instance_name]
            component = self._component(instance_name)
            outputs = component._block_mode_response(
                inputs, self._local_parameters[instance_name]
            )
            self._check_time_axis(instance_name, outputs)
            self.component_outputs[instance_name] = {
                port: signal
                for port, signal in outputs.items()
                if not _is_backward_port(component, port)
            }
            self.forward_reflections[instance_name] = {
                port: signal
                for port, signal in outputs.items()
                if _is_backward_port(component, port)
            }

        if self.simulation_parameters.backward_pass:
            self._run_backward_pass()

        input_signals = {}
        output_signals = {}
        backward_signals = {}
        for (
            tracked_port_name,
            tracked_port_designator,
        ) in self._instantiated_circuit.port_lookup_table.items():
            instance_name, port_name = tracked_port_designator.split(",")
            if port_name in self.component_inputs[instance_name]:
                input_signals[tracked_port_name] = self.component_inputs[instance_name][
                    port_name
                ]
            if port_name in self.component_outputs[instance_name]:
                output_signals[tracked_port_name] = self.component_outputs[
                    instance_name
                ][port_name]
            for backward in (self.backward_outputs, self.backward_inputs):
                if port_name in backward.get(instance_name, {}):
                    backward_signals[tracked_port_name] = backward[instance_name][
                        port_name
                    ]
        simulation_result = BlockModeSimulationResult(
            input_signals, output_signals, backward_signals
        )
        simulation_result.sample_periods = {
            name: self.rate_schedule.port_domain(*designator.split(",")).dt
            for name, designator in self._instantiated_circuit.port_lookup_table.items()
        }

        logger.debug("Block mode simulation completed in %.6f s", time() - tic)
        return simulation_result

    def _setup_rates(self):
        """Infer each region's sample rate; give every component the
        simulation parameters of its own region (see simphony.circuit.rates)."""
        self.rate_schedule = infer_sample_rates(
            self._instantiated_circuit, self.simulation_parameters
        )
        self._local_parameters = {}
        for instance_name in self.block_mode_order:
            domain = self.rate_schedule.instance_domain(instance_name)
            self._local_parameters[instance_name] = local_parameters(
                self.simulation_parameters, domain
            )
            if instance_name in self.rate_schedule.rate_changers:
                self._component(instance_name).input_parameters = local_parameters(
                    self.simulation_parameters,
                    self.rate_schedule.input_domain(instance_name),
                )

    def _check_time_axis(self, instance_name, outputs):
        """Enforce the block-mode convention: axis 0 of every data field is
        time, with the length of the port's rate region."""
        for port, signal in outputs.items():
            fields = getattr(type(signal), "_data_fields", ())
            expected = self.rate_schedule.port_domain(
                instance_name, port
            ).num_time_steps
            for field_name in fields:
                value = getattr(signal, field_name)
                length = jnp.shape(value)[0] if jnp.ndim(value) else None
                if length != expected:
                    raise ValueError(
                        f"{instance_name},{port}: {type(signal).__name__}.{field_name} has "
                        f"shape {jnp.shape(value)}; block-mode data fields must have "
                        f"time on axis 0 with {expected} samples for this rate region"
                    )

    def _component(self, instance_name):
        return self._instantiated_circuit.instantiated_flat_netlist["instances"][
            instance_name
        ]["model"]

    def _collect_component_inputs(self, component) -> dict:
        inputs = {}
        input_components = [
            u for u, v in self._instantiated_circuit.graph.in_edges(component)
        ]
        for input_component in input_components:
            input_edges = self._instantiated_circuit.graph.get_edge_data(
                input_component, component
            )
            for edge_number, edge in input_edges.items():
                inputs[edge["dst_port"]] = self.component_outputs[input_component][
                    edge["src_port"]
                ]
        self.component_inputs[component] = inputs

    def _run_backward_pass(self):
        """Propagate backward-travelling waves in reverse topological order.

        For each component, the backward waves arriving at its output ports
        are the backward waves leaving the downstream input ports they are
        connected to. The component is evaluated with those arrivals (plus
        the non-optical inputs of the forward pass, e.g. modulator drive
        voltages; forward optical inputs are omitted so forward reflections
        are not counted twice). The backward waves it emits on its input
        ports are added to its forward-pass reflections.
        """
        graph = self._instantiated_circuit.graph
        for instance_name in reversed(self.block_mode_order):
            component = self._component(instance_name)
            arrivals = {}
            for _, downstream, edge in graph.out_edges(instance_name, data=True):
                signal = self.backward_outputs.get(downstream, {}).get(edge["dst_port"])
                if signal is not None:
                    arrivals[edge["src_port"]] = signal
            self.backward_inputs[instance_name] = arrivals

            emitted = dict(self.forward_reflections[instance_name])
            if arrivals:
                inputs = {
                    port: signal
                    for port, signal in self.component_inputs[instance_name].items()
                    if component._port_lookup_table[port].type != "optical"
                }
                inputs.update(arrivals)
                outputs = component._block_mode_response(
                    inputs, self._local_parameters[instance_name]
                )
                for port, signal in outputs.items():
                    if not _is_backward_port(component, port):
                        continue  # Re-reflection into the forward direction
                    if port in emitted:
                        signal = signal.replace(
                            amplitude=signal.amplitude + emitted[port].amplitude
                        )
                    emitted[port] = signal
            self.backward_outputs[instance_name] = emitted

    def _determine_block_mode_order_nx_method(self, instantiated_circuit):
        """Return the directed acyclic execution order for Block mode."""
        try:
            return list(nx.topological_sort(instantiated_circuit.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError(
                "Failed to determine Block mode order: circular dependencies detected"
            )


def _is_backward_port(component, port_name) -> bool:
    """True if a signal on `port_name` leaving `component` travels backward.

    That is the case for optical input ports (not bidirectional ones).
    """
    port = component._port_lookup_table.get(port_name)
    return (
        port is not None and port.type == "optical" and port.directionality == "input"
    )
