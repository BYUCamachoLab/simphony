from .simulation import Simulation, SimulationResult, SimulationParameters
from simphony.circuit.circuit import Circuit
from simphony.component.component import SampleModeComponent
# from simphony.libraries.analytic import advance
# from simphony.simulation.advance import _advance as advance
from simphony.simulation.terminator import ElectricalTerminator, OpticalTerminator, LogicTerminator
# from simphony.simulation import SimulationParameters
from simphony.signal.sample_mode import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal
from simphony.signal.block_mode import BlockModeOpticalSignal, BlockModeElectricalSignal, BlockModeLogicSignal
from functools import partial
from dataclasses import replace

from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from simphony.simulation import jax_tools
from flax import struct

from time import time

from dataclasses import field
from simphony.simulation.simulation import SimulationMode

from simphony.circuit.netlist import generate_unique_string

from typing import Annotated, Optional

# def replace(obj, **updates):
#     fields = obj.__dict__.copy()
#     fields.update(updates)
#     return obj.__class__(**fields)

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
        self.input_signals  = input_signals
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
    use_optimized:
        Enables optimized structured state-space updates where available.
    time_batch_size:
        Optional number of time steps per scan chunk. Component state is
        carried between chunks and tracked outputs are concatenated.
    mode_identifiers:
        Inherited optical mode labels.
    """
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.SAMPLE_MODE)
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jax.numpy.array([1.55e-6]))
    directed: bool = False
    dt: float = 1e-14
    num_time_steps: int = 50
    use_optimized: bool = True
    time_batch_size: Optional[int] = None
    # random_seed = 0

@struct.dataclass
class SampleModeSimulationState():
    """Mutable global state carried through a sample-mode scan."""
    prng_key: Annotated[jax.Array, "shape=(2,), dtype=jax.uint32"]=field(default_factory=lambda: jax.random.PRNGKey(0))

class SampleModeOpticalTerminator(OpticalTerminator, SampleModeComponent):
    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_state, simulation_parameters: SampleModeSimulationParameters):
        """Compute the next state of the system."""
        return {"out": SampleModeOpticalSignal(amplitude=jnp.zeros((1, len(simulation_parameters.mode_identifiers)), dtype=complex), wavelength=jnp.array([1.55e-6]))}, state

class SampleModeElectricalTerminator(ElectricalTerminator, SampleModeComponent):
    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_state, simulation_parameters: SampleModeSimulationParameters):
        """Compute the next state of the system."""
        return {"out": SampleModeElectricalSignal(voltage=0.0)}, state
    
class SampleModeLogicTerminator(LogicTerminator, SampleModeComponent):
    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_state, simulation_parameters: SampleModeSimulationParameters):
        """Compute the next state of the system."""
        return {"out": SampleModeLogicSignal(voltage=0)}, state


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
        simulation_parameters = None,
        # ports = None,
        # circuit: Circuit,
        # ports = None
    ):

        if settings is None:
            settings = {}
        if simulation_parameters is None:
            simulation_parameters = SampleModeSimulationParameters()
        if tracked_ports is None:
            tracked_ports = circuit.netlist["top_level"]['ports']

        self.simulation_parameters = simulation_parameters
        self.circuit = deepcopy(circuit)
        self.insert_terminators()
        
        # self.flat_circuit = circuit.flatten()
        self.settings = deepcopy(settings)
        self.tracked_ports = deepcopy(tracked_ports)
        self.component_inputs = {}
        self.component_outputs = {}

    # def __init__(
    #     self, 
    #     circuit: Circuit
    # ):
    #     self._validate_circuit(circuit)
    #     circuit = self._insert_terminations(circuit)
    #     # circuit = self._insert_advance_blocks(circuit) # Our method of delay compensation
    #     self.circuit = self._make_all_connections_bidirectional(circuit)
    #     self.reset_settings(use_default_settings=True)

    #     self._instance_names = list(self.circuit.graph.nodes)
        
    #     self._instance_ports = []
    #     for inst_name in self._instance_names:
    #         component = self.circuit.netlist['instances'][inst_name]['component']
    #         self.circuit.models[component]
    #         model = self.circuit.models[component]
    #         ports = model.optical_ports + model.electrical_ports + model.logic_ports
    #         ports.sort()
            
    #         self._instance_ports += [(inst_name, port) for port in ports]

    def run(
        self,
        use_jit = True,
    ) -> SampleModeSimulationResult:
        """Run the sample-mode simulation.

        Parameters
        ----------
        use_jit:
            If true, use `jax.lax.scan`; otherwise use the Python scan helper,
            which is easier to debug.

        Returns
        -------
        dict
            Current implementation returns the nested scan output structure
            keyed by instance and port for each time step.
        """
        return self._run_single(use_jit=use_jit, time_batch_size=self.simulation_parameters.time_batch_size)

    def _run_single(
        self,
        use_jit=True,
        time_batch_size=None,
    ) -> SampleModeSimulationResult:
        # Currently, we pass in randomly generated prng keys through the simualtion parameters field, so
        # we have to get rid of the enum field to make jax happy.
        # otherwise, I would simply mark the dataclass as static
        # sim_mode = self.simulation_parameters.simulation_mode
        # self.simulation_parameters = self.simulation_parameters.replace(simulation_mode=str(sim_mode))

        self._instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters, tracked_ports=self.tracked_ports, directed=False)
        self._predecessors_map, self._successors_map = self.edge_lookup_tables()
        port_lookup_table = self._instantiated_circuit.port_lookup_table
        
        # self.output_optical_port_lookup_table = {}
        # for instance_name, instance_data in self._instantiated_circuit.instantiated_flat_netlist["instances"].items():
        #     instance_output_port_lut = instance_data['model']._output_port_lookup_table
        #     self.output_optical_port_lookup_table[instance_name] = {p.name for p in instance_output_port_lut.values if (p.directionality)}
        
        N = self.simulation_parameters.num_time_steps

        self.components = {}
        for instance_name, instance_data in self._instantiated_circuit.instantiated_flat_netlist['instances'].items():
            self.components[instance_name] = instance_data['model']

        initial_states = {}
        for instance_name, instance in self.components.items():
            initial_states[instance_name] = instance._sample_mode_initial_state(self.simulation_parameters)

        # Build lightweight maps from tracked port name → (instance, port) for both
        # output and input sides.  These are passed as static partial arguments so
        # _system_step can emit only the tracked signals instead of the full circuit
        # output dict, avoiding O(N_steps × N_instances × N_ports) memory allocation.
        tracked_output_map = {}
        tracked_input_map  = {}
        for tracked_port_name, tracked_port_designator in port_lookup_table.items():
            inst, port = tracked_port_designator.split(",", 1)
            tracked_output_map[tracked_port_name] = (inst, port)
            for src_inst, src_port in self._predecessors_map.get((inst, port), []):
                tracked_input_map[tracked_port_name] = (src_inst, src_port)
                break

        current_outputs = self._initial_outputs()
        tic = time()
        simulation_state = SampleModeSimulationState(prng_key=jax.random.PRNGKey(self.simulation_parameters.seed))
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
        print(toc - tic)

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
        """Attach terminator source components to unconnected input-like ports."""
        unconnected_ports = self.circuit.unconnected_ports(inputs_only=True)
        optical_port_terminator_number = 0
        electrical_port_terminator_number = 0
        logic_port_terminator_number = 0
        instance_separator = generate_unique_string(self.circuit.netlist['top_level']['instances'].keys())
        model_separator = generate_unique_string(self.circuit.models.keys())
        for instance_name, _ports in unconnected_ports.items():
            for unconnected_port in _ports:
                if unconnected_port.type == "optical":
                    terminator_instance_name = f"optical_terminator{instance_separator}{optical_port_terminator_number}"
                    terminator_model_name = f"optical_terminator_{model_separator}"
                    self.circuit.add_component(terminator_instance_name, terminator_model_name, SampleModeOpticalTerminator)
                    self.circuit.add_connection(terminator_instance_name, "out", instance_name, unconnected_port.name)
                    optical_port_terminator_number += 1
                elif unconnected_port.type == "electrical":
                    terminator_instance_name = f"electrical_terminator{instance_separator}{electrical_port_terminator_number}"
                    terminator_model_name = f"electrical_terminator_{model_separator}"
                    self.circuit.add_component(terminator_instance_name, terminator_model_name, SampleModeElectricalTerminator)
                    self.circuit.add_connection(terminator_instance_name, "out", instance_name, unconnected_port.name)
                    electrical_port_terminator_number += 1
                elif unconnected_port.type == "logic":
                    terminator_instance_name = f"logic_terminator{instance_separator}{logic_port_terminator_number}"
                    terminator_model_name = f"logic_terminator_{model_separator}"
                    self.circuit.add_component(terminator_instance_name, terminator_model_name, SampleModeLogicTerminator)
                    self.circuit.add_connection(terminator_instance_name, "out", instance_name, unconnected_port.name)
                    logic_port_terminator_number += 1
    
    def edge_lookup_tables(self):
        """Build predecessor and successor lookup tables keyed by instance port."""
        successors_map = {}
        predecessors_map = {}
        instances = self._instantiated_circuit.instantiated_flat_netlist['instances']
    
        for inst_name, inst_data in instances.items():
            for port in inst_data['model'].ports:
                successors_map[(inst_name, port.name)] = []
                predecessors_map[(inst_name, port.name)] = []

        for inst_name in instances.keys():
            in_edges = self._instantiated_circuit.graph.in_edges(inst_name, data=True)
            for src_node, dst_node, data in in_edges:
                src_port = data['src_port']
                dst_port = data['dst_port']
                predecessors_map[(dst_node, dst_port)].append((src_node, src_port))
            
            out_edges = self._instantiated_circuit.graph.out_edges(inst_name, data=True)
            for src_node, dst_node, data in out_edges:
                src_port = data['src_port']
                dst_port = data['dst_port']
                successors_map[(src_node, src_port)].append((dst_node, dst_port))

        # Last 

        return predecessors_map, successors_map

    def _system_step(self, carry, x, simulation_parameters=None,
                     tracked_output_map=None, tracked_input_map=None):
        system_outputs = carry[0]
        states = carry[1]
        simulation_state = carry[2]
        prng_key = simulation_state.prng_key

        system_inputs = {}
        for instance_name, instance in self.components.items():
            system_inputs[instance_name] = self._get_inputs(instance_name, system_outputs)

        for instance_name, instance in self.components.items():
            prng_key, subkey = jax.random.split(prng_key)
            simulation_state = replace(simulation_state, prng_key=subkey)
            inputs = system_inputs[instance_name]
            input_state = states[instance_name]
            instance_outputs, output_state = instance._sample_mode_step(inputs, input_state, simulation_state, simulation_parameters)
            states[instance_name] = output_state
            system_outputs[instance_name] = system_outputs[instance_name] | instance_outputs

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
        # ports = self.components[instance_name].optical_ports + self.components[instance_name].electrical_ports + self.components[instance_name].logic_ports
        # OPTICAL_NULL_SRC_NODE = 0
        # OPTICAL_NULL_SRC_PORT = 0
        for port in ports:
            # Sample mode simulations do not support multiple inputs
            # Assumed list length is 1
            for src_node, src_port in self._predecessors_map[(instance_name, port.name)]:
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
                    initial_outputs[inst_name][port_name] = SampleModeOpticalSignal(amplitude, wl)
                elif port.type == "electrical":
                    voltage = 0.0
                    initial_outputs[inst_name][port_name] = SampleModeElectricalSignal(voltage)
                elif port.type == "logic":
                    value = 0
                    initial_outputs[inst_name][port_name] = SampleModeLogicSignal(value)
            # for o_port in model.optical_ports:
            #     amplitude = jnp.zeros((optical_wavelengths.shape[0], 1), dtype=complex)
            #     wl = optical_wavelengths
            #     initial_outputs[inst_name][o_port] = SampleModeOpticalSignal(amplitude, wl)
            # for e_port in model.electrical_ports:
            #     initial_outputs[inst_name][e_port] = SampleModeElectricalSignal(0)
            # for l_port in model.logic_ports:
            #     value = 0
            #     initial_outputs[inst_name][l_port] = SampleModeLogicSignal(value)

        return initial_outputs

    # def _make_all_connections_bidirectional(self, circuit):
    #     new_models = circuit.models
    #     netlist = circuit.netlist
    #     new_instances = deepcopy(netlist['instances'])
    #     new_ports = deepcopy(netlist['ports'])
    #     new_connections = deepcopy(netlist['connections'])

    #     for src, dst in netlist['connections'].items():
    #         destinations = [s.strip() for s in dst.split(';') if s]
    #         for new_source in destinations:
    #             new_destination = src
    #             previous_destinations = ''
    #             if new_source in netlist['connections']:
    #                 previous_destinations = netlist['connections'][new_source] + ";"
    #             new_connections[new_source] = previous_destinations + f'{new_destination}'
        
    #     new_netlist = {
    #         'instances': new_instances,
    #         'connections': new_connections,
    #         'ports': new_ports
    #     }

    #     new_circuit = Circuit(new_netlist, new_models)
    #     return new_circuit
    
    # def _insert_terminations(self, circuit):
    #     netlist = circuit.netlist
    #     new_instances = deepcopy(netlist['instances'])
    #     new_connections = deepcopy(netlist['connections'])
    #     new_ports = deepcopy(netlist['ports'])

    #     unterminated_ports = set()
    #     for instance_name in netlist['instances'].keys():
    #         component = circuit.graph.nodes[instance_name]['component']
    #         model = circuit.models[component]
    #         all_instance_ports = set(model.optical_ports + model.electrical_ports + model.logic_ports)
    #         terminated_instance_ports = set()
            
    #         in_edges = circuit.graph.in_edges(instance_name, data=True)
    #         out_edges = circuit.graph.out_edges(instance_name, data=True)
    #         for _, dst, data in in_edges:
    #             # _ = data['src_port']
    #             dst_port = data['dst_port']
    #             terminated_instance_ports.add(dst_port)
            
    #         for src, _, data in out_edges:
    #             src_port = data['src_port']
    #             # dst_port = data['dst_port']
    #             terminated_instance_ports.add(src_port)
    #             pass

    #         unterminated_instance_ports = all_instance_ports - terminated_instance_ports
    #         for unterminated_port_name in unterminated_instance_ports:
    #             unterminated_ports.add((instance_name, unterminated_port_name))
        
    #     termination_numbers = {
    #         'optical': 0,
    #         'electrical': 0,
    #         'logic': 0,
    #     }
    #     termination_components = set()
    #     for instance_name, port_name in unterminated_ports:
    #         termination_type = circuit.get_port_type(instance_name, port_name)
    #         termination_component = f'_{termination_type}_termination'
    #         termination_components.add((termination_component, termination_type))
    #         termination_inst = f'{termination_component}{termination_numbers[termination_type]}'
    #         new_instances[termination_inst] = {
    #             'component': termination_component,
    #             'settings': {},
    #         }

    #         ###
    #         # TODO: ADD THE CONNECTION
    #         ###
    #         new_connections[termination_inst+",out"] = instance_name + ',' + port_name
    #         termination_numbers[termination_type] += 1
    
    #     new_models = deepcopy(circuit.models)
    #     for termination_component, termination_type in termination_components:
    #         new_models[termination_component] = termination(termination_type=termination_type)
        
    #     new_netlist = {
    #         'instances': new_instances,
    #         'connections': new_connections,
    #         'ports': new_ports
    #     }
        
    #     new_circuit = Circuit(new_netlist, new_models)
    #     return new_circuit

    # def _insert_advance_blocks(self, circuit):
    #     netlist = circuit.netlist
    #     new_instances = deepcopy(netlist['instances'])
    #     new_connections = {}
    #     new_ports = deepcopy(netlist['ports'])

    #     advance_numbers = {
    #         'optical': 0,
    #         'electrical': 0,
    #         'logic': 0,
    #     }

    #     advance_components = set()

    #     for src, dst in netlist['connections'].items():
    #         src_inst, src_port = src.split(',')
    #         connection_type = circuit.get_port_type(src_inst, src_port)
    #         advance_component = f'_{connection_type}_advance'
    #         advance_components.add((advance_component, connection_type))
    #         advance_inst = f'{advance_component}{advance_numbers[connection_type]}'
    #         new_instances[advance_inst] = {
    #             'component': advance_component,
    #             'settings': {},
    #         }

    #         new_connections[src] = advance_inst + ',in'
    #         new_connections[advance_inst + ',out'] = dst
    #         advance_numbers[connection_type] += 1
    
    #     new_models = deepcopy(circuit.models)
    #     for advance_component, advance_type in advance_components:
    #         new_models[advance_component] = advance(advance_type=advance_type)
        
    #     new_netlist = {
    #         'instances': new_instances,
    #         'connections': new_connections,
    #         'ports': new_ports
    #     }

        
    #     new_circuit = Circuit(new_netlist, new_models)
    #     return new_circuit
        
    # def _validate_circuit(self, circuit: Circuit):
    #     for component_name in circuit.graph.nodes:
    #         model_name = circuit.netlist['instances'][component_name]['component']
    #         model = circuit.models[model_name]
    #         if not issubclass(model, SampleModeComponent):
    #             raise ValueError(f"{model} is NOT a SampleModeComponent")

    #     # TODO: Check that each connection is one port to one port
