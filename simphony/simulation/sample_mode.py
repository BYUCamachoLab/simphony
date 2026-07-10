import logging
from copy import deepcopy
from dataclasses import field, replace
from functools import partial
from time import time
from typing import Annotated, Optional

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from simphony.circuit.circuit import Circuit
from simphony.circuit.netlist import generate_unique_string
from simphony.component.component import SampleModeComponent
from simphony.component.port import Port
from simphony.signal.sample_mode import (
    SampleModeElectricalSignal,
    SampleModeLogicSignal,
    SampleModeOpticalSignal,
)
from simphony.simulation import jax_tools
from simphony.simulation.simulation import SimulationMode
from simphony.simulation.terminator import (
    ElectricalTerminator,
    LogicTerminator,
    OpticalTerminator,
)

from .simulation import Simulation, SimulationParameters, SimulationResult

logger = logging.getLogger(__name__)


def _terminator_ports(port_type: str, directionality: str):
    return [Port(name="out", type=port_type, directionality=directionality)]


class SampleModeSimulationResult(SimulationResult):
    """Signals collected from a completed sample-mode simulation.

    Attributes
    ----------
    input_signals:
        Signals received at each tracked port (the predecessor's output,
        one step earlier than the component's own output).
    output_signals:
        Signals emitted by the component at each tracked port across all
        time steps.
    """

    def __init__(self, input_signals: dict, output_signals: dict):
        self.input_signals = input_signals
        self.output_signals = output_signals


@struct.dataclass
class SampleModeSimulationParameters(SimulationParameters):
    """Global settings for a sample-mode simulation.

    Sample mode advances the circuit one time sample at a time while preserving
    per-component state between samples.

    Attributes
    ----------
    optical_baseband_wavelengths:
        Carrier wavelengths, in meters, tracked by sample-mode optical signals.
    dt:
        Time step, in seconds.
    num_time_steps:
        Number of sample updates to run.
    use_state_space_optimization:
        Enables optimized structured state-space updates where available.
    time_batch_size:
        Optional number of time steps per scan chunk. Component state is
        carried between chunks and tracked outputs are concatenated.
    mode_identifiers:
        Inherited optical mode labels.
    """

    simulation_mode: SimulationMode = field(
        default_factory=lambda: SimulationMode.SAMPLE_MODE
    )
    optical_baseband_wavelengths: jax.Array = field(
        default_factory=lambda: jax.numpy.array([1.55e-6])
    )
    directed: bool = False
    dt: float = 1e-14
    num_time_steps: int = 50
    use_state_space_optimization: bool = True
    time_batch_size: Optional[int] = None
    # random_seed = 0


@struct.dataclass
class SampleModeSimulationState:
    """Mutable global state carried through a sample-mode scan."""

    prng_key: Annotated[jax.Array, "shape=(2,), dtype=jax.uint32"] = field(
        default_factory=lambda: jax.random.PRNGKey(0)
    )


class SampleModeOpticalTerminator(OpticalTerminator, SampleModeComponent):
    def sample_mode_step(
        self,
        inputs: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        """Compute the next state of the system."""
        wavelengths = simulation_parameters.optical_baseband_wavelengths
        return {
            "out": SampleModeOpticalSignal(
                amplitude=jnp.zeros(
                    (wavelengths.shape[0], len(simulation_parameters.mode_identifiers)),
                    dtype=complex,
                ),
                wavelength=wavelengths,
            )
        }, state


class SampleModeOpticalSinkTerminator(SampleModeOpticalTerminator):
    ports = _terminator_ports("optical", "input")


class SampleModeBidirectionalOpticalTerminator(SampleModeOpticalTerminator):
    ports = _terminator_ports("optical", "bidirectional")


class SampleModeElectricalTerminator(ElectricalTerminator, SampleModeComponent):
    def sample_mode_step(
        self,
        inputs: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        """Compute the next state of the system."""
        return {"out": SampleModeElectricalSignal(voltage=0.0)}, state


class SampleModeElectricalSinkTerminator(SampleModeElectricalTerminator):
    ports = _terminator_ports("electrical", "input")


class SampleModeBidirectionalElectricalTerminator(SampleModeElectricalTerminator):
    ports = _terminator_ports("electrical", "bidirectional")


class SampleModeLogicTerminator(LogicTerminator, SampleModeComponent):
    def sample_mode_step(
        self,
        inputs: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        """Compute the next state of the system."""
        return {"out": SampleModeLogicSignal(voltage=0)}, state


class SampleModeLogicSinkTerminator(SampleModeLogicTerminator):
    ports = _terminator_ports("logic", "input")


class SampleModeBidirectionalLogicTerminator(SampleModeLogicTerminator):
    ports = _terminator_ports("logic", "bidirectional")


_TERMINATOR_MODEL_BY_PORT = {
    ("optical", "input"): SampleModeOpticalTerminator,
    ("optical", "output"): SampleModeOpticalSinkTerminator,
    ("optical", "bidirectional"): SampleModeBidirectionalOpticalTerminator,
    ("electrical", "input"): SampleModeElectricalTerminator,
    ("electrical", "output"): SampleModeElectricalSinkTerminator,
    ("electrical", "bidirectional"): SampleModeBidirectionalElectricalTerminator,
    ("logic", "input"): SampleModeLogicTerminator,
    ("logic", "output"): SampleModeLogicSinkTerminator,
    ("logic", "bidirectional"): SampleModeBidirectionalLogicTerminator,
}


class SampleModeSimulation(Simulation):
    """Run a circuit one time sample at a time.

    Sample mode is intended for components that inherit from the SampleModeComponent class that expose
    `sample_mode_initial_state` and `sample_mode_step`. The simulator inserts
    terminators for unconnected input-like ports, instantiates the circuit,
    initializes component state, then advances all components for
    `num_time_steps`.

    Parameters
    ----------
    circuit:
        Circuit to simulate.
    settings:
        Per-instance constructor settings.
    tracked_ports:
        Optional mapping of names to `"instance,port"` designators. Defaults to
        the circuit top-level ports.
    simulation_parameters:
        Shared `SampleModeSimulationParameters`. Defaults are used when omitted.
    """

    def __init__(
        self,
        circuit: Circuit,
        settings,
        tracked_ports: dict = None,
        simulation_parameters=None,
    ):
        if settings is None:
            settings = {}
        if simulation_parameters is None:
            simulation_parameters = SampleModeSimulationParameters()
        if tracked_ports is None:
            tracked_ports = circuit.netlist["top_level"]["ports"]

        self.simulation_parameters = simulation_parameters
        self.circuit = deepcopy(circuit)
        self.insert_terminators()
        self.settings = deepcopy(settings)
        self.tracked_ports = deepcopy(tracked_ports)
        self.component_inputs = {}
        self.component_outputs = {}

    def run(
        self,
        use_jit=True,
    ) -> SampleModeSimulationResult:
        """Run the sample-mode simulation.

        Parameters
        ----------
        use_jit:
            If true, use the production `jax.lax.scan` path. Passing false uses
            `simphony.simulation.jax_tools.python_based_scan`, which is
            debug-only and should not be used for production simulation results.

        Returns
        -------
        SampleModeSimulationResult
            Tracked input and output signal histories keyed by tracked-port
            name.
        """
        return self._run_single(
            use_jit=use_jit, time_batch_size=self.simulation_parameters.time_batch_size
        )

    def _run_single(
        self,
        use_jit=True,
        time_batch_size=None,
    ) -> SampleModeSimulationResult:
        self.component_inputs = {}
        self.component_outputs = {}

        # Currently, we pass in randomly generated prng keys through the simulation parameters field, so
        # we have to get rid of the enum field to make jax happy.
        # otherwise, I would simply mark the dataclass as static
        # sim_mode = self.simulation_parameters.simulation_mode
        # self.simulation_parameters = self.simulation_parameters.replace(simulation_mode=str(sim_mode))

        self._instantiated_circuit = self.circuit.instantiate(
            self.settings,
            self.simulation_parameters,
            tracked_ports=self.tracked_ports,
            directed=False,
        )
        self._predecessors_map, self._successors_map = self.edge_lookup_tables()
        port_lookup_table = self._instantiated_circuit.port_lookup_table

        N = self.simulation_parameters.num_time_steps

        self.components = {}
        for (
            instance_name,
            instance_data,
        ) in self._instantiated_circuit.instantiated_flat_netlist["instances"].items():
            self.components[instance_name] = instance_data["model"]

        initial_states = {}
        for instance_name, instance in self.components.items():
            initial_states[instance_name] = instance._sample_mode_initial_state(
                self.simulation_parameters
            )

        # Build lightweight maps from tracked port name to (instance, port) for both
        # output and input sides.  These are passed as static partial arguments so
        # _system_step can emit only the tracked signals instead of the full circuit
        # output dict, avoiding O(N_steps * N_instances * N_ports) memory allocation.
        tracked_output_map = {}
        tracked_input_map = {}
        for tracked_port_name, tracked_port_designator in port_lookup_table.items():
            inst, port = tracked_port_designator.split(",", 1)
            tracked_output_map[tracked_port_name] = (inst, port)
            for src_inst, src_port in self._predecessors_map.get((inst, port), []):
                tracked_input_map[tracked_port_name] = (src_inst, src_port)
                break

        current_outputs = self._initial_outputs()
        tic = time()
        simulation_state = SampleModeSimulationState(
            prng_key=jax.random.PRNGKey(self.simulation_parameters.seed)
        )
        system_step = partial(
            self._system_step,
            simulation_parameters=self.simulation_parameters,
            tracked_output_map=tracked_output_map,
            tracked_input_map=tracked_input_map,
        )
        carry = (current_outputs, initial_states, simulation_state)
        _, stacked_tracked = self._run_time_batches(
            system_step=system_step,
            carry=carry,
            num_time_steps=N,
            time_batch_size=time_batch_size,
            use_jit=use_jit,
        )
        toc = time()
        logger.debug("Sample mode simulation completed in %.6f s", toc - tic)

        # stacked_tracked is already keyed by tracked port name with shape (N, L, M).
        return SampleModeSimulationResult(
            input_signals=stacked_tracked["inputs"],
            output_signals=stacked_tracked["outputs"],
        )

    def _make_scan_runner(self, system_step, chunk_length, use_jit):
        if use_jit:

            def run_scan(carry):
                return lax.scan(system_step, carry, length=chunk_length)

            return jax.jit(run_scan)

        def run_python_scan(carry):
            return jax_tools.python_based_scan(system_step, carry, length=chunk_length)

        return run_python_scan

    def _run_time_batches(
        self,
        system_step,
        carry,
        num_time_steps,
        time_batch_size=None,
        use_jit=True,
    ):
        if time_batch_size is None or int(time_batch_size) >= num_time_steps:
            run_scan = self._make_scan_runner(system_step, num_time_steps, use_jit)
            return run_scan(carry)

        time_batch_size = int(time_batch_size)
        if time_batch_size <= 0:
            raise ValueError("time_batch_size must be a positive integer")

        full_chunks, remainder = divmod(num_time_steps, time_batch_size)
        run_full_chunk = self._make_scan_runner(system_step, time_batch_size, use_jit)
        run_remainder = (
            self._make_scan_runner(system_step, remainder, use_jit)
            if remainder
            else None
        )

        tracked_chunks = []
        for _ in range(full_chunks):
            carry, tracked_chunk = run_full_chunk(carry)
            tracked_chunks.append(tracked_chunk)

        if run_remainder is not None:
            carry, tracked_chunk = run_remainder(carry)
            tracked_chunks.append(tracked_chunk)

        return carry, self._combine_time_batch_pytrees(tracked_chunks)

    def _combine_time_batch_pytrees(self, tracked_chunks):
        if not tracked_chunks:
            return {"inputs": {}, "outputs": {}}

        return jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *tracked_chunks,
        )

    def insert_terminators(self):
        """Attach terminators to all unconnected signal ports."""
        unconnected_ports = self.circuit.unconnected_ports(inputs_only=False)
        terminator_numbers = {"optical": 0, "electrical": 0, "logic": 0}
        instance_separator = generate_unique_string(
            self.circuit.netlist["top_level"]["instances"].keys()
        )
        model_separator = generate_unique_string(self.circuit.models.keys())
        for instance_name, _ports in unconnected_ports.items():
            for unconnected_port in _ports:
                terminator_model = _TERMINATOR_MODEL_BY_PORT.get(
                    (unconnected_port.type, unconnected_port.directionality)
                )
                if terminator_model is None:
                    continue

                terminator_instance_number = terminator_numbers[unconnected_port.type]
                terminator_instance_name = (
                    f"{unconnected_port.type}_terminator"
                    f"{instance_separator}{terminator_instance_number}"
                )
                terminator_model_name = (
                    f"{unconnected_port.type}_terminator_for_"
                    f"{unconnected_port.directionality}_{model_separator}"
                )
                self.circuit.add_component(
                    terminator_instance_name,
                    terminator_model_name,
                    terminator_model,
                )

                if unconnected_port.directionality == "output":
                    self.circuit.add_connection(
                        instance_name,
                        unconnected_port.name,
                        terminator_instance_name,
                        "out",
                    )
                else:
                    self.circuit.add_connection(
                        terminator_instance_name,
                        "out",
                        instance_name,
                        unconnected_port.name,
                    )

                terminator_numbers[unconnected_port.type] += 1

    def edge_lookup_tables(self):
        """Build predecessor and successor lookup tables keyed by instance
        port."""
        successors_map = {}
        predecessors_map = {}
        instances = self._instantiated_circuit.instantiated_flat_netlist["instances"]

        for inst_name, inst_data in instances.items():
            for port in inst_data["model"].ports:
                successors_map[(inst_name, port.name)] = []
                predecessors_map[(inst_name, port.name)] = []

        for inst_name in instances.keys():
            in_edges = self._instantiated_circuit.graph.in_edges(inst_name, data=True)
            for src_node, dst_node, data in in_edges:
                src_port = data["src_port"]
                dst_port = data["dst_port"]
                predecessors_map[(dst_node, dst_port)].append((src_node, src_port))

            out_edges = self._instantiated_circuit.graph.out_edges(inst_name, data=True)
            for src_node, dst_node, data in out_edges:
                src_port = data["src_port"]
                dst_port = data["dst_port"]
                successors_map[(src_node, src_port)].append((dst_node, dst_port))

        return predecessors_map, successors_map

    def _system_step(
        self,
        carry,
        x,
        simulation_parameters=None,
        tracked_output_map=None,
        tracked_input_map=None,
    ):
        system_outputs = carry[0]
        states = carry[1]
        simulation_state = carry[2]
        prng_key = simulation_state.prng_key

        system_inputs = {}
        for instance_name, instance in self.components.items():
            system_inputs[instance_name] = self._get_inputs(
                instance_name, system_outputs
            )

        for instance_name, instance in self.components.items():
            prng_key, subkey = jax.random.split(prng_key)
            simulation_state = replace(simulation_state, prng_key=subkey)
            inputs = system_inputs[instance_name]
            input_state = states[instance_name]
            instance_outputs, output_state = instance._sample_mode_step(
                inputs, input_state, simulation_state, simulation_parameters
            )
            states[instance_name] = output_state
            system_outputs[instance_name] = (
                system_outputs[instance_name] | instance_outputs
            )

        new_carry = (system_outputs, states, simulation_state)

        # Emit only the tracked-port signals.  The full system_outputs remains in
        # the carry for routing but is never stacked across time steps, keeping
        # scan memory proportional to the number of tracked ports rather than to
        # the total number of ports in the circuit.
        y = {
            "outputs": {
                name: system_outputs[inst][port]
                for name, (inst, port) in tracked_output_map.items()
            },
            "inputs": {
                name: system_outputs[inst][port]
                for name, (inst, port) in tracked_input_map.items()
            },
        }
        return new_carry, y

    def _get_inputs(self, instance_name, current_outputs):
        inputs = {}
        ports = self.components[instance_name].ports
        for port in ports:
            # Sample mode simulations do not support multiple inputs
            # Assumed list length is 1
            for src_node, src_port in self._predecessors_map[
                (instance_name, port.name)
            ]:
                inputs[port.name] = current_outputs[src_node][src_port]

        return inputs

    def _initial_outputs(self):
        wl = self.simulation_parameters.optical_baseband_wavelengths
        modes = self.simulation_parameters.mode_identifiers
        initial_outputs = {}
        for inst_name, model in self.components.items():
            initial_outputs[inst_name] = {}
            for port_name, port in model._port_lookup_table.items():
                if port.type == "optical":
                    amplitude = jnp.zeros((wl.shape[0], len(modes)), dtype=complex)
                    initial_outputs[inst_name][port_name] = SampleModeOpticalSignal(
                        amplitude, wl
                    )
                elif port.type == "electrical":
                    voltage = 0.0
                    initial_outputs[inst_name][port_name] = SampleModeElectricalSignal(
                        voltage
                    )
                elif port.type == "logic":
                    value = 0
                    initial_outputs[inst_name][port_name] = SampleModeLogicSignal(value)

        return initial_outputs
