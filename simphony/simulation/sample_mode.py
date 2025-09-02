from .simulation import Simulation, SimulationResult, SimulationParameters
from simphony.circuit import Circuit, SampleModeComponent
# from simphony.libraries.analytic import advance
from simphony.simulation.advance import _advance as advance
from simphony.simulation.termination import _termination as termination
# from simphony.simulation import SimulationParameters
from simphony.signals import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal, BlockModeOpticalSignal, BlockModeElectricalSignal, BlockModeLogicSignal

from dataclasses import replace

from copy import deepcopy
import jax
import jax.numpy as jnp
from jax import lax
from simphony.simulation import jax_tools
from flax import struct

from time import time

from dataclasses import field

# def replace(obj, **updates):
#     fields = obj.__dict__.copy()
#     fields.update(updates)
#     return obj.__class__(**fields)

class SampleModeSimulationResult(SimulationResult):
    def __init__(self):
        pass

@struct.dataclass
class SampleModeSimulationParameters(SimulationParameters):
    # def __init__(
    #     self,
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jnp.array([1.51e-6, 1.52e-6, 1.53e-6, 1.54e-6, 1.55e-6, 1.56e-6, 1.57e-6, 1.58e-6, 1.59e-6]))
    electrical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jnp.array([0]))
    #     **kwargs,
    # ):
    #     super().__init__(**kwargs)
    #     self.optical_baseband_wavelengths = optical_baseband_wavelengths
    #     self.electrical_baseband_wavelengths = electrical_baseband_wavelengths


class SampleModeSimulation(Simulation):
    def __init__(self, circuit: Circuit):
        self._validate_circuit(circuit)
        circuit = self._insert_terminations(circuit)
        # circuit = self._insert_advance_blocks(circuit) # Our method of delay compensation
        self.circuit = self._make_all_connections_bidirectional(circuit)
        self.reset_settings(use_default_settings=True)

        self._instance_names = list(self.circuit.graph.nodes)
        
        self._instance_ports = []
        for inst_name in self._instance_names:
            component = self.circuit.netlist['instances'][inst_name]['component']
            self.circuit.models[component]
            model = self.circuit.models[component]
            ports = model.optical_ports + model.electrical_ports + model.logic_ports
            ports.sort()
            
            self._instance_ports += [(inst_name, port) for port in ports]
        
        self._predecessors_map, self._successors_map = self.edge_lookup_tables()
        
    
    def edge_lookup_tables(self):
        successors_map = {}
        predecessors_map = {}
        for inst_name, port in self._instance_ports:
            successors_map[(inst_name, port)] = []
            predecessors_map[(inst_name, port)] = []

        for inst_name in self._instance_names:
            in_edges = self.circuit.graph.in_edges(inst_name, data=True)
            for src_node, dst_node, data in in_edges:
                src_port = data['src_port']
                dst_port = data['dst_port']
                predecessors_map[(dst_node, dst_port)].append((src_node, src_port))
            
            out_edges = self.circuit.graph.out_edges(inst_name, data=True)
            for src_node, dst_node, data in out_edges:
                src_port = data['src_port']
                dst_port = data['dst_port']
                successors_map[(src_node, src_port)].append((dst_node, dst_port))
        return predecessors_map, successors_map

    def run(
        self, 
        settings: dict = None,
        tracked_ports: dict = None,
        simulation_parameters: SampleModeSimulationParameters = SampleModeSimulationParameters(),
        use_jit: bool = True
        # optical_wavelengths: jax.Array = jnp.asarray([1.55e-6]),
        # electrical_wavelengths: jax.Array = jnp.asarray([0]),
        # sampling_period: float = 1e-15,
        # num_time_steps: int = 10000,
    ) -> SampleModeSimulationResult:
        N = simulation_parameters.num_time_steps
        optical_wavelengths = simulation_parameters.optical_baseband_wavelengths
        electrical_wavelengths = simulation_parameters.electrical_baseband_wavelengths

        if tracked_ports is None:
            tracked_ports = self.circuit.netlist['ports']
        self.tracked_ports = tracked_ports
        self.tracked_signals = {}
        
        for key, value in self.tracked_ports.items():
            self.tracked_signals[key] = {}
            # TODO: Determine port type
            port_type = 'optical'
            if port_type == 'optical':
                A_t = jnp.zeros((N), dtype=complex)
                self.tracked_signals[key]['input'] = BlockModeOpticalSignal(amplitude=A_t.reshape((N, 1, 1)), wavelength=optical_wavelengths)
                self.tracked_signals[key]['output'] = BlockModeOpticalSignal(amplitude=A_t.reshape((N, 1, 1)), wavelength=optical_wavelengths)
            elif port_type == 'electrical':
                A_t = jnp.zeros((N), dtype=complex)
                self.tracked_signals[key]['input'] = BlockModeElectricalSignal(amplitude=A_t.reshape((N, 1)), wavelength=electrical_wavelengths)
                self.tracked_signals[key]['output'] = BlockModeElectricalSignal(amplitude=A_t.reshape((N, 1)), wavelength=electrical_wavelengths)
            elif port_type == 'logic':
                value = jnp.zeros((N), dtype=int)
                self.tracked_signals[key]['input'] = BlockModeLogicSignal(value=value, wavelength=electrical_wavelengths)
                self.tracked_signals[key]['output'] = BlockModeLogicSignal(value=value, wavelength=electrical_wavelengths)
        
        self.reset_settings(use_default_settings=True)
        self.add_settings(settings)
        optical_wavelengths = jnp.sort(optical_wavelengths)
        electrical_wavelengths = jnp.sort(electrical_wavelengths)
        self._instantiate_components(self.settings)
        
        # use_jit = True
        if use_jit:
            self._scan = lax.scan
        else:
            self._scan = jax_tools.python_based_scan

        # Determine the maximum delay compensation
        max_delay_compensation = 0
        for _, instance in self.components.items():
            if instance.delay_compensation > max_delay_compensation:
                max_delay_compensation = instance.delay_compensation
        
        initial_states = {}
        for instance_name, instance in self.components.items():
            initial_states[instance_name] = instance._sample_mode_initial_state(
                simulation_parameters,
            )
            # initial_states[instance_name] = instance._initial_state()

        current_outputs = self._initial_outputs(optical_wavelengths, electrical_wavelengths)
        time_steps = jnp.arange(0, N, 1, dtype=int)
        tic = time()
        _, system_outputs = self._scan(self._system_step, (current_outputs, initial_states, simulation_parameters), length=N)
        toc = time()
        elapsed_time = toc - tic
        return system_outputs

    def _system_step(self, carry, x):
        time_step = x
        system_outputs = carry[0]
        states = carry[1]
        simulation_parameters = carry[2]
        prng_key = simulation_parameters.prng_key
        # y = self.tracked_signals

        old_system_outputs = system_outputs
        system_inputs = {}
        for instance_name, instance in self.components.items():
            system_inputs[instance_name] = self._get_inputs(instance_name, system_outputs)

        for instance_name, instance in self.components.items():
            # Generate a unique key for each time_step/instance
            prng_key, subkey = jax.random.split(prng_key)
            simulation_parameters = replace(simulation_parameters,prng_key=subkey)

            inputs = system_inputs[instance_name]
            input_state = states[instance_name]
            instance_outputs, output_state = instance._sample_mode_step(inputs, input_state, simulation_parameters)
            states[instance_name] = output_state
            system_outputs[instance_name] = system_outputs[instance_name] | instance_outputs

            
        new_carry = (system_outputs, states, simulation_parameters)
        y = system_outputs
        return new_carry, y
    
    def _get_inputs(self, instance_name, current_outputs):
        inputs = {}
        ports = self.components[instance_name].optical_ports + self.components[instance_name].electrical_ports + self.components[instance_name].logic_ports
        # OPTICAL_NULL_SRC_NODE = 0
        # OPTICAL_NULL_SRC_PORT = 0
        for port in ports:
            # Sample mode simulations do not support multiple inputs
            # Assumed list length is 1
            src = self._predecessors_map[(instance_name, port)]
            src_node, src_port = src[0]
            inputs[port] = current_outputs[src_node][src_port]

            pass

        return inputs

    def _initial_outputs(self, optical_wavelengths, electrical_wavelengths):
        initial_outputs = {}
        for inst_name, model in self.components.items():
            initial_outputs[inst_name] = {}
            for o_port in model.optical_ports:
                amplitude = jnp.zeros((optical_wavelengths.shape[0], 1), dtype=complex)
                wl = optical_wavelengths
                initial_outputs[inst_name][o_port] = SampleModeOpticalSignal(amplitude, wl)
            for e_port in model.electrical_ports:
                voltage = jnp.zeros((electrical_wavelengths.shape[0]), dtype=complex)
                wl = electrical_wavelengths
                initial_outputs[inst_name][e_port] = SampleModeElectricalSignal(voltage, wl)
            for l_port in model.logic_ports:
                value = 0
                initial_outputs[inst_name][l_port] = SampleModeLogicSignal(value)

        return initial_outputs

    def _make_all_connections_bidirectional(self, circuit):
        new_models = circuit.models
        netlist = circuit.netlist
        new_instances = deepcopy(netlist['instances'])
        new_ports = deepcopy(netlist['ports'])
        new_connections = deepcopy(netlist['connections'])

        for src, dst in netlist['connections'].items():
            destinations = [s.strip() for s in dst.split(';') if s]
            for new_source in destinations:
                new_destination = src
                previous_destinations = ''
                if new_source in netlist['connections']:
                    previous_destinations = netlist['connections'][new_source] + ";"
                new_connections[new_source] = previous_destinations + f'{new_destination}'
        
        new_netlist = {
            'instances': new_instances,
            'connections': new_connections,
            'ports': new_ports
        }

        new_circuit = Circuit(new_netlist, new_models)
        return new_circuit
    
    def _insert_terminations(self, circuit):
        netlist = circuit.netlist
        new_instances = deepcopy(netlist['instances'])
        new_connections = deepcopy(netlist['connections'])
        new_ports = deepcopy(netlist['ports'])

        unterminated_ports = set()
        for instance_name in netlist['instances'].keys():
            component = circuit.graph.nodes[instance_name]['component']
            model = circuit.models[component]
            all_instance_ports = set(model.optical_ports + model.electrical_ports + model.logic_ports)
            terminated_instance_ports = set()
            
            in_edges = circuit.graph.in_edges(instance_name, data=True)
            out_edges = circuit.graph.out_edges(instance_name, data=True)
            for _, dst, data in in_edges:
                # _ = data['src_port']
                dst_port = data['dst_port']
                terminated_instance_ports.add(dst_port)
            
            for src, _, data in out_edges:
                src_port = data['src_port']
                # dst_port = data['dst_port']
                terminated_instance_ports.add(src_port)
                pass

            unterminated_instance_ports = all_instance_ports - terminated_instance_ports
            for unterminated_port_name in unterminated_instance_ports:
                unterminated_ports.add((instance_name, unterminated_port_name))
        
        termination_numbers = {
            'optical': 0,
            'electrical': 0,
            'logic': 0,
        }
        termination_components = set()
        for instance_name, port_name in unterminated_ports:
            termination_type = circuit.get_port_type(instance_name, port_name)
            termination_component = f'_{termination_type}_termination'
            termination_components.add((termination_component, termination_type))
            termination_inst = f'{termination_component}{termination_numbers[termination_type]}'
            new_instances[termination_inst] = {
                'component': termination_component,
                'settings': {},
            }

            ###
            # TODO: ADD THE CONNECTION
            ###
            new_connections[termination_inst+",out"] = instance_name + ',' + port_name
            termination_numbers[termination_type] += 1
    
        new_models = deepcopy(circuit.models)
        for termination_component, termination_type in termination_components:
            new_models[termination_component] = termination(termination_type=termination_type)
        
        new_netlist = {
            'instances': new_instances,
            'connections': new_connections,
            'ports': new_ports
        }
        
        new_circuit = Circuit(new_netlist, new_models)
        return new_circuit

    def _insert_advance_blocks(self, circuit):
        netlist = circuit.netlist
        new_instances = deepcopy(netlist['instances'])
        new_connections = {}
        new_ports = deepcopy(netlist['ports'])

        advance_numbers = {
            'optical': 0,
            'electrical': 0,
            'logic': 0,
        }

        advance_components = set()

        for src, dst in netlist['connections'].items():
            src_inst, src_port = src.split(',')
            connection_type = circuit.get_port_type(src_inst, src_port)
            advance_component = f'_{connection_type}_advance'
            advance_components.add((advance_component, connection_type))
            advance_inst = f'{advance_component}{advance_numbers[connection_type]}'
            new_instances[advance_inst] = {
                'component': advance_component,
                'settings': {},
            }

            new_connections[src] = advance_inst + ',in'
            new_connections[advance_inst + ',out'] = dst
            advance_numbers[connection_type] += 1
    
        new_models = deepcopy(circuit.models)
        for advance_component, advance_type in advance_components:
            new_models[advance_component] = advance(advance_type=advance_type)
        
        new_netlist = {
            'instances': new_instances,
            'connections': new_connections,
            'ports': new_ports
        }

        
        new_circuit = Circuit(new_netlist, new_models)
        return new_circuit
        
    def _validate_circuit(self, circuit: Circuit):
        for component_name in circuit.graph.nodes:
            model_name = circuit.netlist['instances'][component_name]['component']
            model = circuit.models[model_name]
            if not issubclass(model, SampleModeComponent):
                raise ValueError(f"{model} is NOT a SampleModeComponent")

        # TODO: Check that each connection is one port to one port

