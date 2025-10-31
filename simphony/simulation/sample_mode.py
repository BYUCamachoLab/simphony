from functools import partial
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
import numpy as np
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
    CHUNK_SIZE = 100000
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
    ) -> SampleModeSimulationResult:
        self.CHUNK = self.CHUNK_SIZE
        N = simulation_parameters.num_time_steps
        optical_wavelengths = simulation_parameters.optical_baseband_wavelengths
        electrical_wavelengths = simulation_parameters.electrical_baseband_wavelengths
        self.simulation_parameters = simulation_parameters
        if tracked_ports is None:
            tracked_ports = self.circuit.netlist['ports']
        self.tracked_ports = tracked_ports
        self.reset_settings(use_default_settings=True)
        self.add_settings(settings)
        optical_wavelengths = jnp.sort(optical_wavelengths)
        electrical_wavelengths = jnp.sort(electrical_wavelengths)
        self._instantiate_components(self.settings)
        self.initial_states = {}
        for instance_name, instance in self.components.items():
            self.initial_states[instance_name] = instance._sample_mode_initial_state(simulation_parameters)
            print(f"Initial state for {instance_name} computed")
        self.current_outputs = self._initial_outputs(optical_wavelengths, electrical_wavelengths)
        self._tracked_aliases = tuple(self.tracked_ports.keys())
        self._tracked_pairs = tuple(tuple(x.strip() for x in self.tracked_ports[a].split(",")) for a in self._tracked_aliases)
        self._L = int(optical_wavelengths.shape[0])
 
        if use_jit:
            self._scan = lax.scan
            self._run = jax.jit(self._run_chunk, static_argnums=(4))
        else:
            self._scan = jax_tools.python_based_scan
            self._run = self._run_chunk
        i = jnp.int32(0)
        n_full = N // self.CHUNK
        tail = int(N - n_full * self.CHUNK)
        pieces_in = [[] for _ in self._tracked_aliases]
        pieces_out = [[] for _ in self._tracked_aliases]
        jax.block_until_ready(self.current_outputs)
        print("Starting simulation...")
        tick = time()
        for _ in range(n_full):
                ticker = time()
                self.current_outputs, self.initial_states, self.simulation_parameters, i, ys = self._run(self.current_outputs, self.initial_states, self.simulation_parameters, i, self.CHUNK)
                y = jax.device_get(ys)
                for a in range(len(self._tracked_aliases)):
                    pieces_in[a].append(np.asarray(y[:, a, 0, :, :]))
                    pieces_out[a].append(np.asarray(y[:, a, 1, :, :]))
                tocker = time()
                time_remaining = (tocker-ticker) * (n_full - _)
                print(f"Completed {_+1} / {n_full} chunks Estimated time remaining: {time_remaining}")  
        if tail:
            self.current_outputs, self.initial_states, self.simulation_parameters, i, ys = self._run(self.current_outputs, self.initial_states, self.simulation_parameters, i, tail)
            y = jax.device_get(ys)
            for a in range(len(self._tracked_aliases)):
                pieces_in[a].append(np.asarray(y[:, a, 0, :, :]))
                pieces_out[a].append(np.asarray(y[:, a, 1, :, :]))
        
        tracked = {}
        for a, alias in enumerate(self._tracked_aliases):
            if pieces_in[a]:
                ain = np.concatenate(pieces_in[a], axis=0)
                aout = np.concatenate(pieces_out[a], axis=0)
            else:
                ain = np.zeros((0, self._L, 1), dtype=np.complex128)
                aout = np.zeros((0, self._L, 1), dtype=np.complex128)
            tracked[alias] = {}
            tracked[alias]['input'] = SampleModeOpticalSignal(amplitude=jnp.asarray(ain), wavelength=optical_wavelengths)
            tracked[alias]['output'] = SampleModeOpticalSignal(amplitude=jnp.asarray(aout), wavelength=optical_wavelengths)
        jax.block_until_ready(tracked)
        tock = time()
        print(f"Simulation time: {tock - tick:.3f} s")
        return tracked

    def _run_chunk(self, system_outputs, states, simulation_parameters, i0, steps):
        def step(carry, _):
            system_outputs, states, simulation_parameters, i = carry
            system_inputs = {name: self._get_inputs(name, system_outputs) for name in self.components.keys()}
            new_outputs = {}
            for instance_name, instance in self.components.items():
                subkey = jax.random.fold_in(simulation_parameters.prng_key, i)
                sim_i = replace(simulation_parameters, prng_key=subkey)
                inputs_i = system_inputs[instance_name]
                state_i = states[instance_name]
                outs_i, state_o = instance._sample_mode_step(inputs_i, state_i, sim_i)
                states[instance_name] = state_o
                merged = dict(system_outputs[instance_name])
                merged.update(outs_i)
                new_outputs[instance_name] = merged
            meas = []

            for inst, port in self._tracked_pairs:
                amp_in = system_inputs[inst][port].amplitude[:, 0]
                amp_out = new_outputs[inst][port].amplitude[:, 0]
                meas.append(jnp.stack([amp_in.reshape((self._L, 1)), amp_out.reshape((self._L, 1))], axis=0))

            meas = jnp.stack(meas, axis=0)
            return (new_outputs, states, simulation_parameters, i + jnp.int32(1)), meas
        (system_outputs_f, states_f, simulation_parameters_f, i_f), ys = self._scan(step, (system_outputs, states, simulation_parameters, i0), xs=None, length=steps)
        return system_outputs_f, states_f, simulation_parameters_f, i_f, ys
        
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

