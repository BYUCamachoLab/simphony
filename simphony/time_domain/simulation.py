from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import sax
import jax
import jax.numpy as jnp
import math
from jax import config

from simphony.time_domain.time_system import TimeSystemIIR, TimeSystem
config.update("jax_enable_x64", True)

from simphony.utils import dict_to_matrix
from simphony.time_domain.pole_residue_model import BVF_Options, IIRModelBaseband

from simphony.simulation import Simulation, SimulationResult
from dataclasses import dataclass

from scipy.interpolate import interp1d
from simphony.exceptions import UndefinedActiveComponent
import sys
from tqdm.auto import tqdm



@dataclass
class TimeResult(SimulationResult):
    """
    Stores and manages the results of a time-domain photonic simulation.

    **User-Facing**: Typically, you'll create and obtain an instance of TimeResult
    from the TimeSim.run(...) function, then call plot_sim() if needed.

    Attributes:
        outputs (ArrayLike): Dictionary mapping output port names to their respective time-domain signals.
        t (ArrayLike): Time array used in the simulation.
        inputs (ArrayLike): Dictionary mapping input port names to their respective time-domain signals.
        S_params (ArrayLike): The S-parameter matrix (or list of matrices for sub-circuits) used in the simulation.
    """
    outputs: ArrayLike
    t: ArrayLike
    inputs: ArrayLike
    S_params: ArrayLike

    def plot_sim(self) -> None:
        """
        Plots the intensity of each port's input and output signals over time,
        aligned so that input and output for the same port share a row.
        """
        # Find ports present in both inputs and outputs
        ports = [k for k in self.inputs.keys() if k in self.outputs]
        if not ports:
            raise ValueError("No matching ports found in inputs and outputs.")

        n = len(ports)
        fig, axs = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)

        for i, key in enumerate(ports):
            # input intensity
            axs[i, 0].plot(self.t, jnp.abs(self.inputs[key])**2)
            axs[i, 0].set_title(f'Input Signal {key}')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Intensity')

            # output intensity
            axs[i, 1].plot(self.t, jnp.abs(self.outputs[key])**2)
            axs[i, 1].set_title(f'Output Signal {key}')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('Intensity')

        plt.tight_layout()
        plt.show()


class TimeSim(Simulation):
    """
    A class for time-domain photonic circuit simulation, allowing for both passive
    and active components. 

    **User-Facing**: Typical usage involves:
      1) Initializing with a netlist and component models (`__init__`).
      2) Building the model (`build_model`).
      3) Running the simulation (`run`), which returns a TimeResult object.

    Internally, if active components are detected, the netlist is decomposed into
    passive sub-circuits. Each passive sub-circuit is converted into an IIR time-domain
    model. Active components are then iterated in the time-domain, step-by-step, 
    alongside these passive sub-circuits.

    Attributes:
        netlist (dict): Dictionary describing the devices and how they are interconnected.
        component_models (dict): Dictionary mapping component names to their frequency-domain models.
        active_components (set): Names of active components in the netlist.
    """

    def __init__(self, netlist: dict, models: dict, active_components: set = None):
        """
        Initializes the TimeSim object with a netlist, a dictionary of models,
        and an optional set of active components.

        Args:
            netlist (dict): Dictionary containing 'instances', 'connections', and 'ports'.
            component_models (dict): Dictionary mapping component names to corresponding models.
            active_components (set, optional): Names of active components in the netlist. Defaults to None.
        """
        self.netlist = netlist
        self.models = models
        self.active_components = active_components
        for model in self.models:
            if isinstance(self.models[model], TimeSystem):
                for instance in self.netlist["instances"]:
                    if self.netlist["instances"][instance] == model:
                        if self.active_components is None:
                            self.active_components = set()
                            if instance not in self.active_components:
                                self.active_components.add(instance)
                        else:
                            if instance not in self.active_components:
                                self.active_components.add(instance)
        
        # Extract netlist info for convenience
        self.instances = netlist["instances"]
        self.connections = netlist["connections"]
        self.ports = netlist["ports"]

        # Internal placeholders for models, S-parameters, and time stepping
        self.dt = None
        self.passive_subnetlists = None
        self.removed_connections = None
        self.removed_ports = None
        self.passive_reconnections = None
        self.S_params_dict = None

        # Holds the final time-domain netlist after re-wiring sub-circuits and active comps
        self.td_netlist = {"models": {}, "connections": {}, "ports": {}}

        # Sub-circuit time-domain models (passive only)
        self.subcircuit_time_systems = []

        # For storing signals during step-by-step simulation
        self.inputs = {}
        self.outputs = {}
        self.instance_outputs = {}
        self.t = None

    def build_model(
        self,
        wvl: np.ndarray = np.linspace(1.5, 1.6, 200),
        center_wvl: float = 1.55,
        model_order: int = 50,
        model_parameters: dict = None,
        dt: float = 1e-14,
        max_size: int = 10,
        suppress_output: bool = False
    ) -> None:
        """
        Builds or configures the underlying IIR model(s) for the circuit.

        **User-Facing**: This is typically the second step (after __init__).
        If active components are present, the netlist is decomposed into 
        passive sub-circuits, each turned into an IIR model. Active components 
        connect to these sub-circuits at designated ports.

        Args:
            wvl (np.ndarray): Wavelength array for frequency-domain model generation.
            center_wvl (float): Center wavelength for frequency shift.
            model_order (int): The order of the IIR approximation.
            model_parameters (dict): Parameters to pass into the frequency-domain model evaluation.
            dt (float): Simulation time step.
            max_size (int): Maximum size of any strongly-connected sub-circuit (for splitting).
            suppress_output (bool): If True, suppresses printing of intermediate results.
        """
        self.dt = dt
        self.max_size = max_size
        c_light = 299792458
        center_freq = c_light / (center_wvl * 1e-6)
        freqs = c_light / (wvl * 1e-6) - center_freq

        # Beta is used for the broadband vector fitting (BVF)
        sampling_freq = -1 / dt
        beta = sampling_freq / (freqs[-1] - freqs[0])
        bvf_options = BVF_Options(beta=beta)
        self.suppress_output = suppress_output

        # If active components exist, break out passive sub-circuits
        if self.active_components is not None:
            (
                self.passive_subnetlists,
                self.removed_connections,
                self.removed_ports,
            ) = self.create_passive_sub_netlists(
                self.instances,
                self.connections,
                self.ports,
                self.active_components,
            )
            if not self.suppress_output:
            # Print the passive sub-netlists for debugging/logging
                for i, sub_net in enumerate(self.passive_subnetlists):
                    print(f"\n--- Passive Sub-Netlist {i} ---")
                    print("\nInstances:", sub_net["instances"])
                    print("\nConnections:", sub_net["connections"])
                    print("\nPorts:", sub_net["ports"])
                    print()

            # Build frequency-domain circuits via SAX and convert to time-domain models
            sub_circuit_list = {}
            port_map_list = {}

            try:
                for i, sub_net in enumerate(self.passive_subnetlists):
                    # Create circuit with sax
                    circuittemp, _ = sax.circuit(
                        netlist=sub_net,
                        models=self.models,
                    )
                    sub_circuit_list[i] = circuittemp
                    port_map_list[i] = sub_net['ports']

            except TypeError as originalerror:
                # This generally happens if an active component is incorrectly included
                # in a subcircuit that should be purely passive
                raise UndefinedActiveComponent(
                    "Active component is included in a passive-only circuit"
                ) from originalerror

            # Evaluate S-parameters and build time-domain IIR models
            self.S_params_dict = {}
            for i, circuit in sub_circuit_list.items():
                s_params_dict = circuit(**model_parameters)
                s_matrix = np.asarray(dict_to_matrix(s_params_dict))
                self.S_params_dict[i] = s_matrix

                sorted_ports = sorted(port_map_list[i].keys())
                
                iir_model = IIRModelBaseband(
                    wvl,
                    center_wvl,
                    s_matrix,
                    model_order,
                    options=bvf_options
                )
                td_system = TimeSystemIIR(iir_model, sorted_ports)
                self.subcircuit_time_systems.append(td_system)
                
            # Connect the newly created sub-circuits and the active components 
            # in a consolidated time-domain netlist (self.td_netlist)
            self.prepare_time_domain_netlist(port_map_list)

        else:
            circuit, _ = sax.circuit(netlist=self.netlist, models=self.models)
            fd_params = circuit(**model_parameters)
            s_matrix = np.asarray(dict_to_matrix(fd_params))
            self.S_params_dict = s_matrix

            single_iir_model = IIRModelBaseband(
                wvl,
                center_wvl,
                s_matrix,
                model_order,
                options=bvf_options
            )
            self.time_system = TimeSystemIIR(single_iir_model)

    def run(self, t: ArrayLike, input_signals: dict, reset: bool = True) -> TimeResult:
        """
        Runs the time-domain simulation over a specified time array with given input signals.

        **User-Facing**: This is typically the final step. Provide your time array
        and input signals here.

        Args:
            t (ArrayLike): Original time array.
            input_signals (dict): Dictionary of input signals keyed by port name.
            reset (bool): If True, resets the internal state of all sub-circuits and active components.

        Returns:
            TimeResult: An object containing the outputs, time array, inputs, and S-parameters.
        """
        self.t = t
        self.inputs = input_signals
        

        # Interpolate input signals to the time resolution defined by self.dt
        self.t, self.inputs = self.interpolate_inputs()
        n_steps   = len(self.t)
        dtype_out = jnp.complex128            # or jnp.float32 if you store intensities

        
        if reset:
            for model in self.subcircuit_time_systems:
                model.reset()

        # If active components exist, do multi-sub-circuit stepping
        if self.active_components is not None:
            # Initialize top-level outputs
            self.outputs = {p: jnp.zeros((n_steps,), dtype=dtype_out)
                        for p in self.td_netlist["ports"]}        
            
            
            # Initialize instance outputs for both active devices and sub-circuits
            for instance_name, td_model in self.td_netlist["models"].items():
                self.instance_outputs[instance_name] = {
                    port: jnp.array([0 + 0j]) for port in td_model.ports
                }
            i = 0
            # Step through time, one index at a time
            for time_index, _ in tqdm(enumerate(self.t) ):
                self.step(time_index)
                # print(f"Time index: {i} / {len(self.t)}")
                i += 1
              
        # If no active components, do a single time-domain response (already built)
        else:
            self.outputs, _ = self.time_system.response(self.inputs, time_sim=False)

        return TimeResult(
            outputs=self.outputs,
            t=self.t,
            inputs=self.inputs,
            S_params=self.S_params_dict
        )

    ############################################################################
    #                         INTERNAL METHODS BELOW                            #
    #     (Users typically do not need to modify or call these directly.)      #
    ############################################################################

    def interpolate_inputs(self) -> tuple:
        """
        Resamples all input signals from the original time array to the new time array 
        defined by self.dt, using linear interpolation.

        Returns:
            tuple: (resampled_time, resampled_inputs)
        """
        if self.dt is None:
            raise ValueError("Time step (dt) not set. Did you forget to call build_model?")

        # Create the new time domain array based on dt
        t_new = np.arange(self.t[0], self.t[-1] + self.dt, self.dt)

        new_inputs = {}
        for key, values in self.inputs.items():
            interp_func = interp1d(self.t, values, kind='linear', fill_value="extrapolate")
            new_inputs[key] = interp_func(t_new)

        return t_new, new_inputs

    def step(self, time_index: int) -> None:
        """
        Steps the entire time-domain netlist forward by one time index.

        Internally collects inputs for each sub-circuit or active component
        from either another component's output or a top-level input signal.
        Then updates the outputs accordingly.

        Args:
            time_index (int): Index in the simulation time array.
        """
        # For each model (passive sub-circuit or active device) in the netlist
        for instance_name, td_system in self.td_netlist["models"].items():
            instance_inputs = {}

            # Gather inputs from connections or from top-level inputs
            for port in td_system.ports:
                check_str = f'{instance_name},{port}'
                source_found = False

                for conn_key, conn_val in self.td_netlist["connections"].items():
                    if check_str == conn_key:
                        src_inst, src_port = conn_val.split(',')
                        instance_inputs[port] = self.instance_outputs[src_inst][src_port]
                        source_found = True
                        break
                    elif check_str == conn_val:
                        src_inst, src_port = conn_key.split(',')
                        instance_inputs[port] = self.instance_outputs[src_inst][src_port]
                        source_found = True
                        break

                # If not found in connections, check if it's a top-level input
                if not source_found:
                    circuit_port = next(
                        (p for p, val in self.td_netlist["ports"].items()
                         if val.split(',')[1].strip() == port),
                        None
                    )
                    if circuit_port is None:
                        # No direct connection, no top-level input => assume zero
                        instance_inputs[port] = jnp.array([0j])
                    else:
                        instance_inputs[port] = jnp.array([self.inputs[circuit_port][time_index]])

            # Active device or passive sub-circuit: get the time-domain response
            if instance_name in self.active_components:
                # If it's an active component, the system might be storing internal states
                outputs = td_system.response(instance_inputs)
            else:
                # Passive sub-circuit with a direct call to .response(...) in TimeSystemIIR
                outputs, _ = td_system.response(instance_inputs)

            # Update instance outputs
            for out_port_name, out_signal in outputs.items():
                self.instance_outputs[instance_name][out_port_name] = out_signal

        # Finally, gather outputs for top-level ports
        for circuit_port, inst_port_str in self.td_netlist["ports"].items():
            inst_name, port_name = inst_port_str.split(',')
            inst_name = inst_name.strip()
            port_name = port_name.strip()
            new_val = self.instance_outputs[inst_name][port_name]
            self.outputs[circuit_port] = jax.lax.dynamic_update_slice(
                    self.outputs[circuit_port],
                    new_val,                      # shape (1,)
                    (time_index,)                 # write at this position
                )

    def prepare_time_domain_netlist(self, port_map_list: dict) -> None:
        """
        Constructs self.td_netlist to unify all passive sub-circuits and active
        components in a single structure for time stepping.

        Adds references to:
          - Passive sub-circuit models in subcircuit_time_systems
          - Active components from the original netlist
          - Ports and connections that link them
        """
        # 1) Insert references to active device models 
        #    (obtained from self.component_models) for removed connections
        for conn_tuple in self.removed_connections:
            left_str, right_str = conn_tuple
            left_comp = left_str.split(',')[0]
            right_comp = right_str.split(',')[0]

            if left_comp in self.active_components:
                if left_comp not in self.td_netlist["models"]:
                    # Add the active model
                    active_model_instance = self.instances.get(left_comp)
                    self.td_netlist["models"][left_comp] = self.models.get(active_model_instance)

            if right_comp in self.active_components:
                if right_comp not in self.td_netlist["models"]:
                    # Add the active model
                    active_model_instance = self.instances.get(right_comp)
                    self.td_netlist["models"][right_comp] = self.models.get(active_model_instance)

        # 2) Insert the newly built passive sub-circuit time-domain models
        for idx, td_system in enumerate(self.subcircuit_time_systems):
            self.td_netlist["models"][f"{idx}"] = td_system

        # 3) Re-link connections and ports between sub-circuits and active components
        for idx, ports_dict in port_map_list.items():
            for local_port_label, global_port_str in ports_dict.items():
                # Check if this local_port_label was part of removed/active edges
                # or if it needs to be a top-level port
                subckt_key = f"{idx},{local_port_label}"

                # If global_port_str was among removed active edges, re-connect
                if any(
                    global_port_str in pair for pair in self.removed_connections
                ):
                    for left_str, right_str in self.removed_connections:
                        if left_str == global_port_str:
                            self.add_to_time_domain_netlist(connection=(subckt_key, right_str))
                        elif right_str == global_port_str:
                            self.add_to_time_domain_netlist(connection=(subckt_key, left_str))
                else:
                    # Otherwise, treat this as a top-level port
                    self.add_to_time_domain_netlist(port=(local_port_label, subckt_key))

        if self.removed_ports is not None:
            for port_label, comp_port_str in self.removed_ports.items():
                self.add_to_time_domain_netlist(port=(port_label, comp_port_str))
        # 4) Add the top-level ports to the netlis
        if not self.suppress_output:
            # (Optional) Print the final time-domain netlist for debugging
            print("\n--- Final Time-Domain Netlist ---")
            print("\nModels:", self.td_netlist["models"])
            print("\nConnections:", self.td_netlist["connections"])
            print("\nPorts:", self.td_netlist["ports"])
            print()

    def add_to_time_domain_netlist(
        self,
        model_name: str = None,
        model_type=None,
        connection: tuple = None,
        port: tuple = None
    ) -> dict:
        """
        Inserts new items into self.td_netlist, which includes:
          - A new time-domain model (model_name, model_type)
          - A new connection (connection[0] -> connection[1])
          - A new external port (port_label -> "instance,port")

        Returns the updated netlist dictionary (self.td_netlist).
        """
        # Add a new model, if provided
        if model_name and model_type:
            self.td_netlist["models"][model_name] = model_type

        # Add a new connection, if provided
        if connection and len(connection) == 2:
            src, dst = connection
            self.td_netlist["connections"][src] = dst

        # Add a new port, if provided
        if port and len(port) == 2:
            port_label, instance_port_str = port
            self.td_netlist["ports"][port_label] = instance_port_str

        return self.td_netlist

    ############################
    # NETLIST DECOMPOSITION    #
    ############################

    def create_passive_sub_netlists(
        self,
        instances: dict,
        connections: dict,
        ports: dict,
        active_components: set,
        directed: bool = False
    ) -> tuple:
        """
        Decomposes the netlist into purely passive sub-netlists by:
          1) Removing edges that touch active components.
          2) Removing top-level ports referencing active components.
          3) Building a component-level graph.
          4) Finding strongly connected components (SCCs) of that graph (splitting large SCCs).
          5) Building a sub-netlist for each SCC.

        Returns:
            tuple:
                sub_netlists (list): A list of sub-netlists (each sub-netlist is a dict).
                removed_connections (list): Connections removed due to active components.
                removed_ports (dict): Ports removed because they referenced active components.
                passive_reconnections (list): Info for reconnecting passive components externally.
        """
        # 1) Remove edges that touch active components
        filtered_connections, removed_active_edges, removed_conns = self.remove_active_edges_and_track_them(
            connections, active_components
        )

            # 2) Remove ports that reference active components
        filtered_ports, removed_ports = self.remove_ports_to_active(ports, active_components)

        # 3) Build a graph for the remaining passive components
        graph = self.build_component_graph_active(
            filtered_connections,
            ports,
            active_components,
            removed_active_edges,
            directed=directed
        )
        # 4) Find strongly connected components (SCCs)
        connected_components = self.tarjan_scc(graph)

        sub_netlists = []
        

        # 5) Build a sub-netlist for each SCC
        for comp_list in connected_components:
            comp_set = set(comp_list)
            sub_net = self.build_sub_netlist(
                comp_set,
                filtered_connections,
                filtered_ports,
                removed_conns,
                removed_ports,
                instances,
                active_components
            )
            if sub_net["instances"]:
                sub_netlists.append(sub_net)
                
        return sub_netlists, removed_conns, removed_ports

        

        # 4) Find strongly connected components, splitting if they exceed `max_size`
        
    def remove_active_edges_and_track_them(
        self,
        connections: dict,
        active_components: set
    ) -> tuple:
        """
        Removes any connection involving at least one active component.

        Returns:
            tuple:
              (filtered_connections, removed_edges, removed_connections)

            where:
              - filtered_connections is the subset of connections that remain purely passive.
              - removed_edges is a list of (passive_component, port) that connected to active components.
              - removed_connections is the list of removed connections as full strings.
        """
        filtered_connections = {}
        removed_edges = []
        removed_connections = []

        for k, v in connections.items():
            compA, _ = k.split(',')
            compB, _ = v.split(',')

            A_is_active = (compA in active_components)
            B_is_active = (compB in active_components)

            if A_is_active and B_is_active:
                # Both are active => remove it entirely
                continue

            if A_is_active and not B_is_active:
                removed_connections.append((k, v))
                removed_edges.append((compB, v.split(',')[1]))
            elif B_is_active and not A_is_active:
                removed_connections.append((k, v))
                removed_edges.append((compA, k.split(',')[1]))
            else:
                # purely passive => keep
                filtered_connections[k] = v

        return filtered_connections, removed_edges, removed_connections

    def remove_ports_to_active(
        self,
        ports: dict,
        active_components: set
    ) -> tuple:
        """
        Removes top-level ports that directly reference an active component.

        Returns:
            (filtered_ports, removed_ports)
        """
        filtered_ports = {}
        removed_ports = {}

        for port_label, comp_port_str in ports.items():
            comp, _ = comp_port_str.split(',')
            if comp in active_components:
                removed_ports[port_label] = comp_port_str
            else:
                filtered_ports[port_label] = comp_port_str

        return filtered_ports, removed_ports 

    def build_component_graph_active(
        self,
        connections: dict,
        ports: dict,
        active_components: set,
        removed_edges: list,
        directed: bool = False
    ) -> dict:
        """
        Builds a graph (adjacency list) at the component level from the netlist.

        Args:
            connections (dict): e.g. {'compA,portA': 'compB,portB'}.
            ports (dict): top-level ports in the netlist.
            active_components (set): Names of active components.
            removed_edges (list): Edges that were removed because they connect to active components.
            directed (bool): Whether to consider the graph as directed or undirected.

        Returns:
            dict: A dictionary {component: [neighbors, ...]} describing the graph.
        """
        graph = {}

        def add_edge(a, b):
            if a not in graph:
                graph[a] = []
            graph[a].append(b)

        # Add edges for each connection
        for k, v in connections.items():
            compA, _ = k.split(',')
            compB, _ = v.split(',')
            add_edge(compA, compB)
            if not directed:
                add_edge(compB, compA)

        # Ensure each passive component or removed edge is in the graph (even if no connections)
        for _, comp_port_str in ports.items():
            comp, _ = comp_port_str.split(',')
            if comp not in graph and comp not in active_components:
                graph[comp] = []

        for edge_comp, _ in removed_edges:
            if edge_comp not in active_components and edge_comp not in graph:
                graph[edge_comp] = []

        return graph

    def tarjan_scc(self, graph: dict) -> list:
        """
        Computes strongly connected components (SCCs) for the given graph,
        and splits any SCC that exceeds `max_size` by removing selected edges.

        Args:
            graph (dict): adjacency list {node: [neighbors]}.

        Returns:
            list: A list of SCCs, where each SCC is a list of nodes.
        """

        def compute_scc(g: dict) -> list:
            """
            Standard Tarjan's algorithm to compute SCCs without splitting large ones.
            """
            index_counter = [0]
            stack = []
            on_stack = set()
            indices = {}
            lowlinks = {}
            sccs = []

            def strongconnect(node):
                indices[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)

                for w in g.get(node, []):
                    if w not in indices:
                        strongconnect(w)
                        lowlinks[node] = min(lowlinks[node], lowlinks[w])
                    elif w in on_stack:
                        lowlinks[node] = min(lowlinks[node], indices[w])

                if lowlinks[node] == indices[node]:
                    comp_scc = []
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        comp_scc.append(w)
                        if w == node:
                            break
                    sccs.append(comp_scc)

            for n in g.keys():
                if n not in indices:
                    strongconnect(n)
            return sccs
        
        sccs = compute_scc(graph)

        return sccs

    def build_sub_netlist(
    self,
    scc_components,
    all_connections,
    original_ports,
    removed_connections,
    removed_ports,
    instances,
    active_components,
    ):
        """
        Construct a single netlist for the given set of SCC components (e.g. { 'cr1', 'wg1', 'cr2' }).

        1) Keep only those instances in scc_components (all passive).
        2) Keep only those connections that link two components in scc_components.
        3) Keep original ports referencing these components.
        4) For each removed passive->active edge, if the passive comp is in scc_components,
        create a new external port.
        """
        # 1) Filter instances
        scc_instances = {}
        for comp in scc_components:
            if comp in instances:
                scc_instances[comp] = instances[comp]  # e.g. "waveguide", "y_branch", etc.
        
        scc_ports = {}
        # 2) Filter connections
        scc_connections = {}    
        # reconnect_dict = {}   
        
        port_index_counter = 0  # This counter is used for new external port labels.
        
        # 3) Retain original top-level ports that reference a component in the SCC
        for port_label, comp_port_str in original_ports.items():
            comp, _ = comp_port_str.split(',')
            if comp in scc_components:
                scc_ports[port_label] = comp_port_str
                port_index_counter += 1
        temp_index = 0
        for conn_key, conn_value in all_connections.items():
            compA, _ = conn_key.split(',')
            compB, _ = conn_value.split(',')

            if compA in scc_components and compB in scc_components:
                # Connection stays within the SCC
                scc_connections[conn_key] = conn_value

        # 4) Create external ports for removed passive->active edges if the passive component is in scc_components
        aux_port_index = 0  

        for k_string, v_string in removed_connections:
            k_comp = k_string.split(',')[0]
            v_comp = v_string.split(',')[0]
            if k_comp in scc_instances or v_comp in scc_instances:
                if any(k_comp.strip() == value.split(',')[0].strip() for value in removed_ports.values()) and k_comp not in active_components:
                    temp_index = 0
                    if k_comp in removed_ports:
                        for i, j_string in removed_ports.items():
                            j_comp = j_string.split(',')[0]
                            if j_comp == k_comp:
                                new_label = f"o{port_index_counter}"
                                temp_index += 1
                                port_index_counter += 1
                                if new_label in scc_ports:
                                    tempcheck = port_index_counter - 1
                                    while new_label in scc_ports:
                                        new_label = f"o{tempcheck}"
                                        tempcheck -= 1
                                    scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                                else:
                                    scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                                break
                    else:
                        new_label = f"o{port_index_counter}"
                        aux_port_index += 1
                        if new_label in scc_ports:
                            tempcheck = port_index_counter - 1
                            while new_label in scc_ports:
                                new_label = f"o{tempcheck}"
                                tempcheck -= 1
                            scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                        else:
                            scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"

                if v_comp in active_components:
                    temp_index = 0
                    if any(v_comp.strip() == value.split(',')[0].strip() for value in removed_ports.values()):
                        for i, j_string in removed_ports.items():
                            j_comp = j_string.split(',')[0]
                            if j_comp == v_comp:
                                new_label = f"o{port_index_counter}"
                                port_index_counter += 1
                                if new_label in scc_ports:
                                    tempcheck = port_index_counter - 1
                                    while new_label in scc_ports:
                                        new_label = f"o{tempcheck}"
                                        tempcheck -= 1
                                    scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                                else:
                                    scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                                break
                    else:
                        new_label = f"o{port_index_counter}"
                        port_index_counter += 1
                        if new_label in scc_ports:
                            tempcheck = port_index_counter - 1
                            while new_label in scc_ports:
                                new_label = f"o{tempcheck}"
                                tempcheck -= 1
                            scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                        else:
                            scc_ports[new_label] = f"{k_comp},{k_string.split(',')[1]}"
                            
                if k_comp in active_components:
                    temp_index = 0
                    if any(k_comp.strip() == value.split(',')[0].strip() for value in removed_ports.values()):
                        for i, j_string in removed_ports.items():
                            j_comp = j_string.split(',')[0]
                            if j_comp == k_comp:
                                new_label = f"o{port_index_counter}"
                                temp_index += 1
                                port_index_counter += 1
                                if new_label in scc_ports:
                                    tempcheck = port_index_counter - 1
                                    while new_label in scc_ports:
                                        new_label = f"o{tempcheck}"
                                        tempcheck -= 1
                                    scc_ports[new_label] = f"{v_comp},{v_string.split(',')[1]}"
                                else:
                                    scc_ports[new_label] = f"{v_comp},{v_string.split(',')[1]}"
                                break
                    else:
                        new_label = f"o{port_index_counter}"
                        port_index_counter += 1
                        if new_label in scc_ports:
                            tempcheck = port_index_counter - 1
                            while new_label in scc_ports:
                                new_label = f"o{tempcheck}"
                                tempcheck -= 1
                            scc_ports[new_label] = f"{v_comp},{v_string.split(',')[1]}"
                        else:
                            scc_ports[new_label] = f"{v_comp},{v_string.split(',')[1]}"
        
        sub_netlist = {
            "instances": scc_instances,
            "connections": scc_connections,
            "ports": scc_ports,
        }
        return sub_netlist

