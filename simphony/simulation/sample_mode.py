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
from jax import lax
from simphony.simulation import jax_tools
from flax import struct

from time import time

from dataclasses import field
from simphony.simulation.simulation import SimulationMode

from simphony.circuit.netlist import generate_unique_string

from typing import Annotated

# def replace(obj, **updates):
#     fields = obj.__dict__.copy()
#     fields.update(updates)
#     return obj.__class__(**fields)

class SampleModeSimulationResult(SimulationResult):
    def __init__(self):
        pass

@struct.dataclass
class SampleModeSimulationParameters(SimulationParameters):
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.SAMPLE_MODE)
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jax.numpy.array([1.55e-6]))
    directed: bool = False
    dt: float = 1e-14
    num_time_steps: int = 50
    # random_seed = 0

@struct.dataclass
class SampleModeSimulationState():
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
            tracked_ports = circuit.netlist['ports']

        self.simulation_parameters = simulation_parameters
        self.circuit = circuit
        self.insert_terminators()
        
        # self.flat_circuit = circuit.flatten()
        self.settings = settings
        self.tracked_ports = tracked_ports
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
        # Currently, we pass in randomly generated prng keys through the simualtion parameters field, so
        # we have to get rid of the enum field to make jax happy.
        # otherwise, I would simply mark the dataclass as static
        # sim_mode = self.simulation_parameters.simulation_mode
        # self.simulation_parameters = self.simulation_parameters.replace(simulation_mode=str(sim_mode))

        self._instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters, tracked_ports=self.tracked_ports, directed=False)
        self._instantiated_circuit.display()
        self._predecessors_map, self._successors_map = self.edge_lookup_tables()
        port_lookup_table = self._instantiated_circuit.port_lookup_table
        
        # self.output_optical_port_lookup_table = {}
        # for instance_name, instance_data in self._instantiated_circuit.instantiated_flat_netlist["instances"].items():
        #     instance_output_port_lut = instance_data['model']._output_port_lookup_table
        #     self.output_optical_port_lookup_table[instance_name] = {p.name for p in instance_output_port_lut.values if (p.directionality)}
        
        N = self.simulation_parameters.num_time_steps
        optical_wavelengths = self.simulation_parameters.optical_baseband_wavelengths
        
        tracked_signals = {}
        for tracked_port_name, _ in port_lookup_table.items():
            tracked_signals[tracked_port_name] = {}
            # TODO: Determine port type
            port_type = 'optical'
            if port_type == 'optical':
                A_t = jnp.zeros((N), dtype=complex)
                tracked_signals[tracked_port_name]['input'] = BlockModeOpticalSignal(amplitude=A_t.reshape((N, 1, 1)), wavelength=optical_wavelengths)
                tracked_signals[tracked_port_name]['output'] = BlockModeOpticalSignal(amplitude=A_t.reshape((N, 1, 1)), wavelength=optical_wavelengths)
            elif port_type == 'electrical':
                A_t = jnp.zeros((N), dtype=complex)
                tracked_signals[tracked_port_name]['input'] = BlockModeElectricalSignal(voltage=A_t.reshape((N, 1)))
                tracked_signals[tracked_port_name]['output'] = BlockModeElectricalSignal(voltage=A_t.reshape((N, 1)))
            elif port_type == 'logic':
                value = jnp.zeros((N), dtype=int)
                tracked_signals[tracked_port_name]['input'] = BlockModeLogicSignal(value=value)
                tracked_signals[tracked_port_name]['output'] = BlockModeLogicSignal(value=value)
        
        # self.reset_settings(use_default_settings=True)
        # self.add_settings(settings)
        optical_wavelengths = jnp.sort(optical_wavelengths)        
        # use_jit = False
        if use_jit:
            self._scan = lax.scan
        else:
            self._scan = jax_tools.python_based_scan
        
        self.components = {}
        for instance_name, instance_data in self._instantiated_circuit.instantiated_flat_netlist['instances'].items():
            instance = instance_data['model']
            self.components[instance_name] = instance
        
        initial_states = {}
        for instance_name, instance in self.components.items():
            initial_states[instance_name] = instance._sample_mode_initial_state(
                self.simulation_parameters,
            )
            # initial_states[instance_name] = instance._initial_state()


        # current_inputs = self._initial_inputs()
        # current_outputs = deepcopy(current_inputs)
        current_outputs = self._initial_outputs()
        time_steps = jnp.arange(0, N, 1, dtype=int)
        tic = time()
        simulation_state = SampleModeSimulationState(prng_key=jax.random.PRNGKey(self.simulation_parameters.seed))
        system_step = partial(self._system_step, simulation_parameters = self.simulation_parameters)
        _, system_outputs = self._scan(system_step, (current_outputs, initial_states, simulation_state), length=N)
        toc = time()
        elapsed_time = toc - tic
        print(elapsed_time)
        
        return system_outputs

    def insert_terminators(self):
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

    def _system_step(self, carry, x, simulation_parameters=None):
        time_step = x
        system_outputs = carry[0]
        states = carry[1]
        simulation_state = carry[2]
        # simulation_parameters = carry[3]
        prng_key = simulation_state.prng_key
        # y = self.tracked_signals

        old_system_outputs = system_outputs
        system_inputs = {}
        for instance_name, instance in self.components.items():
            system_inputs[instance_name] = self._get_inputs(instance_name, system_outputs)

        for instance_name, instance in self.components.items():
            # Generate a unique key for each time_step/instance
            prng_key, subkey = jax.random.split(prng_key)
            # TODO: Find a more idiomatic place to pass in the subkey (simulation parameters really should be constant values)
            simulation_state = replace(simulation_state,prng_key=subkey)
            
            if "gc1" in instance_name:
                pass
            inputs = system_inputs[instance_name]
            input_state = states[instance_name]
            instance_outputs, output_state = instance._sample_mode_step(inputs, input_state, simulation_state, simulation_parameters)
            states[instance_name] = output_state
            system_outputs[instance_name] = system_outputs[instance_name] | instance_outputs

            
        new_carry = (system_outputs, states, simulation_state)
        y = system_outputs
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

