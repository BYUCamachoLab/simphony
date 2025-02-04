import numpy as np
import matplotlib.pyplot as plt
import sax
from numpy.typing import ArrayLike
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix
from simphony.time_domain.pole_residue_model import IIRModelBaseband
from simphony.time_domain.utils import pole_residue_to_time_system, gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
import re
from simphony.time_domain.ideal import TimePhase_Modulator
from simphony.time_domain.time_system import IIRModelBaseband_to_time_system

def remove_active_edges_and_track_them(connections, active_components):
    """
    Remove any connection that involves an active component.
    Return:
      - filtered_connections (still "compA,portA" -> "compB,portB" form)
      - removed_edges: a list of tuples (passive_comp, active_comp)
        indicating where a passive node used to connect to an active node.
        We'll use this later to create new external ports.
    """
    filtered = {}
    removed_edges = []
    removed_connections = []
    
    for k, v in connections.items():
        compA, portA = k.split(',')
        compB, portB = v.split(',')
        
        A_is_active = (compA in active_components)
        B_is_active = (compB in active_components)
        
        # If both are active, just ignore
        if A_is_active and B_is_active:
            continue
        
        # If exactly one is active, track the passive->active pair
        if A_is_active and not B_is_active:
            # Passive side = compB
            add_tuple = (compB, portB)
            removed_connections.append((k, v))
            removed_edges.append(add_tuple)
            continue
        elif B_is_active and not A_is_active:
            removed_connections.append((k, v))
            add_tuple2 = (compA, portA)
            removed_edges.append(add_tuple2)
            continue
        
        # Otherwise, neither is active => keep this connection
        filtered[k] = v
    
    return filtered, removed_edges, removed_connections


def remove_ports_to_active(ports, active_components):
    """
    Remove top-level ports that directly reference an active component.
    Returns (filtered_ports, removed_ports).
    """
    filtered = {}
    removed = {}
    for port_label, comp_port_str in ports.items():
        comp, port = comp_port_str.split(',')
        if comp in active_components:
            removed[port_label] = comp_port_str
        else:
            filtered[port_label] = comp_port_str
    return filtered, removed


def build_component_graph(connections, ports, directed=False):
    """
    Build a graph (adjacency list) where each node is just the component name.
    For example:
      If "cr1,port_2" -> "wg1,o0" is a connection, we add an edge cr1 -> wg1.
    
    connections: dict { "compA,portA" : "compB,portB" }
    directed: bool (False => treat them as undirected edges)
    Returns: dict { compName : [adjacentCompName, ...], ... }
    """
    graph = {}
    
    def add_edge(a, b):
        if a not in graph:
            graph[a] = []
        graph[a].append(b)
    
    for k, v in connections.items():
        compA, portA = k.split(',')
        compB, portB = v.split(',')
        
        # Add edge from compA -> compB
        add_edge(compA, compB)
        if not directed:
            # Also add compB -> compA
            add_edge(compB, compA)

    for k, v in ports.items():
        comp, port = v.split(',')
        A_is_active = (comp in active_components)
        if comp not in graph and not A_is_active:
            graph[comp] = []
    
    return graph


def tarjan_scc(graph):
    """
    Standard Tarjan's SCC for a graph of the form:
        graph[node] = list of neighbor nodes.
    Returns a list of strongly connected components, e.g. [ [node1,node2,...], [nodeX], ... ].
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
        
        for w in graph.get(node, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[node] = min(lowlinks[node], lowlinks[w])
            elif w in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[w])
        
        # If node is a root node
        if lowlinks[node] == indices[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)
    
    # Run for each node
    for n in graph.keys():
        if n not in indices:
            strongconnect(n)
    
    return sccs


def build_sub_netlist(
    scc_components,
    all_connections,
    original_ports,
    removed_connections,
    removed_ports,
    removed_edges,
    instances
):
    """
    Construct a single netlist for the given set of SCC components (e.g. { 'cr1','wg1','cr2' }).
    
    1) Keep only those instances in scc_components (all passive).
    2) Keep only those connections that link two components in scc_components.
    3) Keep original ports referencing these components.
    4) For each removed passive->active edge, if the passive comp is in scc_components,
       create a new external port.
    """
    # 1) Instances
    scc_instances = {}
    for comp in scc_components:
        if comp in instances:
            scc_instances[comp] = instances[comp]  # e.g. "waveguide", "y_branch", etc.
    
    # 2) Connections
    scc_connections = {}
    for k, v in all_connections.items():
        compA, portA = k.split(',')
        compB, portB = v.split(',')
        if compA in scc_components and compB in scc_components:
            # Connection stays
            scc_connections[k] = v
    total_check = 0
    # 3) Ports (original top-level) - keep only if they reference a component in the SCC
    scc_ports = {}
    for port_label, comp_port_str in original_ports.items():
        comp, port = comp_port_str.split(',')
        if comp in scc_components:
            scc_ports[port_label] = comp_port_str
            total_check += 1
    
    # 4) Add new external ports for each (passiveComp, activeComp) removed edge
    #    if passiveComp is in scc_components
    new_port_index = 0

    
    for k_string, v_string in removed_connections:
        k = k_string.split(',')[0]
        v = v_string.split(',')[0]
        if k in scc_instances or v in scc_instances:
            if any(k in value.split(',')[0] for value in removed_ports.values()):
                temp_index = 0
                if k in removed_ports:
                    for i,j_string in removed_ports.items():
                        j = j_string.split(',')[0]
                        if j== k:
                            new_label = f"o{total_check}"
                            temp_index += 1
                            total_check += 1
                            if new_label in scc_ports:
                                tempcheck = total_check-1
                                while new_label in scc_ports:
                                    new_label = f"o{tempcheck}"
                                    tempcheck-=1
                                scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                            else:
                                scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                            break
                else:
                    new_label = f"o{total_check}"
                    new_port_index += 1
                    
                    if new_label in scc_ports:
                        tempcheck = total_check-1
                        while new_label in scc_ports:
                            new_label = f"o{tempcheck}"
                            tempcheck-=1
                        scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                    else:
                        scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"

            if v in active_components:
                temp_index = 0
                if any(v in value.split(',')[0] for value in removed_ports.values()):
                    for i,j_string in removed_ports.items():
                        j = j_string.split(',')[0]
                        if j== v:
                            new_label = f"o{total_check}"
                            total_check += 1

                            if new_label in scc_ports:
                                tempcheck = total_check-1
                                while new_label in scc_ports:
                                    new_label = f"o{tempcheck}"
                                    tempcheck-=1
                                scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                            else:
                                scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"

                            break
                else:
                    new_label = f"o{total_check}"
                    total_check += 1
                    if new_label in scc_ports:
                        tempcheck = total_check-1
                        while new_label in scc_ports:
                            new_label = f"o{tempcheck}"
                            tempcheck-=1
                        scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                    else:
                        scc_ports[new_label] = f"{k},{k_string.split(',')[1]}"
                    
            
            if k in active_components:
                temp_index = 0
                if any(k in value.split(',')[0] for value in removed_ports.values()):
                    for i,j_string in removed_ports.items():
                        j = j_string.split(',')[0]
                        if j== k:
                            new_label = f"o{total_check}"
                            temp_index += 1
                            total_check += 1
                            if new_label in scc_ports:
                                tempcheck = total_check-1
                                while new_label in scc_ports:
                                    new_label = f"o{tempcheck}"
                                    tempcheck-=1
                                scc_ports[new_label] = f"{v},{v_string.split(',')[1]}"
                            else:
                                scc_ports[new_label] = f"{v},{v_string.split(',')[1]}"
                            
                            break
                else:
                    new_label = f"o{total_check}"
                    total_check += 1
                    if new_label in scc_ports:
                        tempcheck = total_check-1
                        while new_label in scc_ports:
                            new_label = f"o{tempcheck}"
                            tempcheck-=1
                        scc_ports[new_label] = f"{v},{v_string.split(',')[1]}"
                    else:
                        scc_ports[new_label] = f"{v},{v_string.split(',')[1]}"

    new_netlist = {
        "instances": scc_instances,
        "connections": scc_connections,
        "ports": scc_ports,
    }
    return new_netlist


def create_passive_sub_netlists(
    instances, connections, ports, active_components, directed=False
):
    """
    1) Remove edges that touch active components, track them for new external ports.
    2) Build a graph (component-level).
    3) Run SCC.
    4) For each SCC, build a sub-netlist that:
       - has only those passive components
       - has top-level ports referencing them
       - has new "active boundary" ports where an active edge was removed
    5) Return list of such sub-netlists.
    
    Often, if everything is interconnected passively, you'll get 1 SCC.
    """
    # 1) Remove edges to active comps
    filtered_conns, removed_edges, removed_connections = remove_active_edges_and_track_them(connections, active_components)
    
    # 2) Remove top-level ports referencing active comps
    filtered_ports, removed_ports = remove_ports_to_active(ports, active_components)
    
    # 3) Build a graph at component level (ignore port detail)
    graph = build_component_graph(filtered_conns, ports, directed=directed)
    
    # 4) Find SCCs
    sccs = tarjan_scc(graph)  # e.g. [ ['cr1','wg1','cr2'], ['cr3','wg3','cr4'], ... ]
    
    # 5) Build a netlist for each SCC
    sub_netlists = []
    for scc_comp_list in sccs:
        scc_comp_set = set(scc_comp_list)
        # Build the sub-netlist
        sub_nl = build_sub_netlist(
            scc_comp_set,
            filtered_conns,
            filtered_ports,
            removed_connections,
            removed_ports,
            removed_edges,
            instances
        )
        # If the SCC has at least one instance from 'instances', add it
        if sub_nl["instances"]:
            sub_netlists.append(sub_nl)
    
    return sub_netlists, removed_connections, removed_ports


# -----------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # Example netlist
    instances = {
        "wg": "waveguide",
        "y": "y_branch",
        "pm": "phase_modulator",
        "pm2": "phase_modulator",
        "y2": "y_branch",
        "wg2": "waveguide",
        "y3": "y_branch",
        "y4": "y_branch",
        "y5": "y_branch",
        "y6": "y_branch",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "wg5": "waveguide",
        "wg6": "waveguide",
        "bdc": "bidirectional",
        "bdc2": "bidirectional",
        "bdc3": "bidirectional",
    }

    connections = {
        #MZI
        # "pm,o0":"bdc,port_4"
        # "y5,port_1":"y,port_1",
        # "y,port_2": "wg,o0",
        # "y,port_3":"wg2,o0",
        # "y2,port_2":"wg2,o1",
        # "y2,port_3":"wg,o1",

        # "y2,port_1":"y6,port_1",
        # "y6,port_2":"pm,o0",
        # "y6,port_3":"y5,port_3",
        # "y5,port_2":"pm,o1",
        # "y3,port_1":"y5,port_1",

        # "y2,port_1":"pm,o0",
        # "y3,port_1":"pm,o1",

        # "y3,port_2": "wg3,o0",
        # "y3,port_3":"wg4,o0",
        # "y4,port_2":"wg4,o1",
        # "y4,port_3":"wg3,o1",

        # "y4,port_1":"pm2,o0",
        
        # "y4,port_1":"y5,port_1",

        # "y5,port_2": "wg5,o0",
        # "y5,port_3":"wg6,o0",
        # "y6,port_2":"wg6,o1",
        # "y6,port_3":"wg5,o1",

        #Ring Resonator
        # "bdc,port_1":"wg,o1",
        "bdc, port_3": "wg,o0",
        "bdc,port_1": "pm,o1",
        "pm,o0":"wg,o0",
        "bdc,port_3":"wg,o1",
        

        # Coupler
        # "bdc,port_1":"pm,o1",
        
        #Waveguide
        #  "wg,o1":"pm,o0",
        #  "pm,o1":"wg2,o0",

    }

    ports = {
        #MZI
        # "o0": "y5,port_2",
        # "o1": "pm2,o1",
        # "o2": "y5,port_3",
        # "o2": "bdc3,port_3",
        # "o3": "bdc3,port_4",

        # Coupler
        # "o0":"pm,o0",
        # "o1":"bdc,port_2",
        # "o2":"bdc,port_3",
        # "o3":"bdc,port_4",

        #Waveguide
        #  "o0":"wg,o0",
        #  "o1":"pm,o1",
        # "o1":"wg2,o1"

        #Ring Resonator
        "o0":"bdc,port_2",
        "o1":"bdc,port_4"


    }

    active_components = {
        "pm","pm2"
    }
    
    
    def general_function(instances, connections, ports, active_components):
        num_measurements = 200
        model_order = 50
        center_wvl = 1.548  # Center wavelength (µm)
        modelList = []
        N = int(3000)  # Number of time steps
        T = 4e-11
        dt = 1e-14      # Total time duration (40 ps)
        t = jnp.arange(0, T, dt)  # Time array
        t0 = 1e-11  # Pulse start time
        std = 1e-12
        wvl = np.linspace(1.5, 1.6, num_measurements)
        f_mod = 0.0
        m = f_mod * jnp.ones(len(t),dtype = complex)
        timePhaseInstantiated = TimePhase_Modulator(mod_signal=m)
        model_List= {
            "waveguide": siepic.waveguide,
            "y_branch": siepic.y_branch,
            "bidirectional": siepic.bidirectional_coupler,
            "phase_modulator": timePhaseInstantiated,
        }# Build sub-netlists for the passive side
        sub_netlists, removed_connections, removed_ports = create_passive_sub_netlists(
            instances,
            connections,
            ports,
            active_components,
            directed=False  # or True if direction matters
        )
        
        # Usually you'd get 1 sub-netlist if all passive comps are interconnected
        # But if there's more than one SCC, you'll see multiple.
        for i, nl in enumerate(sub_netlists):
            print(f"--- Sub-Netlist {i} ---")
            print("instances:", nl["instances"])
            print("connections:", nl["connections"])
            print("ports:", nl["ports"])
            print()

        sub_circuit_list = {}
        port_list = {}
        for i, netlist in enumerate(sub_netlists):
            dt = 1e-12
            circuittemp, info = sax.circuit(
                netlist=netlist,
                models=model_List,  
            )
            sub_circuit_list[i] = circuittemp
            port_list[i] = netlist['ports']


        

        for i, circuit in sub_circuit_list.items():
            s = circuit(wl = wvl, wg={"length": 50, "loss": 100}, wg2={"length": 10.0, "loss": 100},
                        wg3={"length": 10, "loss": 100}, wg4={"length": 10.0, "loss": 100},
                        wg5={"length": 10, "loss": 100}, wg6={"length": 10.0, "loss": 100},
                        )
            
            S = np.asarray(dict_to_matrix(s))
            temp_port_list = []
            for k,v in port_list[i].items():
                temp_port_list.append(k)
            temp_port_list = sorted(temp_port_list)

            temp_model = IIRModelBaseband_to_time_system(IIRModelBaseband(wvl,center_wvl, S, model_order), temp_port_list)
            modelList.append(temp_model)

        num_outputs = 4
        # inputs = {
        #     f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0 else jnp.zeros_like(t)
        #     for i in range(num_outputs)
        # }
        inputs = {
            f'o{i}': smooth_rectangular_pulse(t,0.5e-11,2.5e-11) if i == 0 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }

        step_list = {
            "step_models":{},
            "step_connections":{},
            "step_ports":{}
        }

        def add_to_netlist(netlist, model_name=None, model_type=None, connection=None, port=None):
            """
            Adds new elements to the netlist.
            
            Args:
                netlist (dict): The existing netlist to update.
                instance_name (str): Name of the new instance to add.
                instance_type (str): Type of the new instance to add (e.g., "ideal_waveguide").
                connection (tuple): Connection to add, e.g., ("instance1,port1", "instance2,port2").
                port (tuple): Port to add, e.g., ("port_name", "instance_name,port").
            
            Returns:
                dict: Updated netlist.
            """
            # Add a new instance, if provided
            if model_name and model_type:
                netlist["step_models"][model_name] = model_type
            
            # Add a new connection, if provided
            if connection and len(connection) == 2:
                netlist["step_connections"][connection[0]] = connection[1]
            
            # Add a new port, if provided
            if port and len(port) == 2:
                netlist["step_ports"][port[0]] = port[1]
            
            return netlist


        for k, v in removed_connections:
            value = v.split(',')[0]
            value2 = k.split(',')[0]
            if value in active_components:
                temp_instance = instances.get(value)
                step_list["step_models"][value] = model_List.get(temp_instance)
            if value2 in active_components:
                temp_instance = instances.get(value2)
                step_list["step_models"][value2] = model_List.get(temp_instance)

        for i, ports in port_list.items():
                    step_list["step_models"][f'{i}'] = modelList[i]

        port_translation_list = []

        for i, ports in port_list.items():
            for j,v in ports.items():
                if any(v in value1 or v in value2 for (value1,value2) in removed_connections):
                    for k, h in removed_connections:
                        if k == v:
                            # port_part = v.split(',')[1]
                            # match = re.search(r'\d+$', port_part)
                            # step_list = add_to_netlist(step_list, connection=(f"{i},o{int(match.group())}", h))

                            step_list = add_to_netlist(step_list, connection=(f"{i},{j}", h))
                        elif h == v:
                            
                            # port_part = v.split(',')[1]
                            # match = re.search(r'\d+$', port_part)
                            step_list = add_to_netlist(step_list, connection=(f"{i},{j}", k))
                            
                else:
                    # port_part = v.split(',')[1]
                    # match = re.search(r'\d+$', port_part)
                    step_list = add_to_netlist(step_list, port=(j,f"{i},{j}"))
                    
                    port_translation_list.append((j,v))

        for k,v in removed_ports.items():
            step_list = add_to_netlist(step_list, port=(k,v))
        
        print(f"--- Netlist ---")
        print("Models:", step_list["step_models"])
        print("connections:", step_list["step_connections"])
        print("ports:", step_list["step_ports"])
        print(port_translation_list)
        print()

        class Stepper:
            def __init__(self, step_list: dict):
                self.step_list = step_list
                
            def run_sim(self, t: ArrayLike, inputs: dict)->dict:
                self.inputs = inputs
                self.instance_outputs = {}
                self.ports = {}
                Statevector_save_list = {}
                for circuit_port, designation in self.step_list['step_ports'].items():
                    instance_name, instance_port = map(str.strip, designation.split(','))
                    self.ports[circuit_port] = (instance_name, instance_port)
                    if instance_name not in active_components:
                        Statevector_save_list[instance_name] = None
                self.outputs = {port: jnp.array([]) for port in self.step_list["step_ports"]}
                for instance_name, time_system in self.step_list["step_models"].items():
                    # if instance_name in active_components:
                    #     self.instance_outputs[instance_name] = {port: }
                    # else:
                    self.instance_outputs[instance_name] = {port: jnp.array([0+0j]) for port in time_system.ports}
                i = 0
                
                
                for _ in t:
                    self.step(i, Statevector_save_list)
                    i +=1
                    pass
                return self.outputs
                
            def step(self, i, Statevector_save_list):
                
                for instance_name,time_system in self.step_list["step_models"].items():
                    instance_inputs = {}

                    for port in time_system.ports:
                        check = f'{instance_name},{port}'
                        found_source = None
                        for k,v in self.step_list["step_connections"].items():
                            if check == k:
                                found_source = 'k'
                                source_name = v.split(',')[0]
                                source_port = v.split(',')[1]
                                instance_inputs[port] = self.instance_outputs[source_name][source_port]
                            elif check == v:
                                found_source = 'v'
                                source_name = k.split(',')[0] 
                                source_port = k.split(',')[1]
                                instance_inputs[port] = self.instance_outputs[source_name][source_port]
                        if found_source == None:
                            designation = f'{port}'
                            circuit_port = next((k for k, v in self.step_list["step_ports"].items() if k == designation), None)
                            instance_inputs[port] = jnp.array([self.inputs[circuit_port][i]])
                            
                    outputs = {}
                    if instance_name in active_components:
                        outputs = time_system.response(instance_inputs)
                    else:
                        outputs,__,state_vector = time_system.response(instance_inputs,state_vector = Statevector_save_list[instance_name])
                        Statevector_save_list[instance_name] = state_vector
                    for port_name in outputs:
                        self.instance_outputs[instance_name][port_name] = outputs[port_name]
                        

                for circuit_port, instance in self.step_list["step_ports"].items():
                    v = instance.split(',')[0]
                    k = instance.split(',')[1]
                    self.outputs[circuit_port] = jnp.concatenate([self.outputs[circuit_port], self.instance_outputs[v][k]])
        

        stepper = Stepper(step_list)
        outputscheck = stepper.run_sim(t, inputs)
        ports = len(outputscheck)
        
        fig, axs = plt.subplots(ports, 2, figsize=(10, 10))
        for i in range(ports):
            axs[i, 0].plot(t, jnp.abs(inputs[f'o{i}'])**2)
            axs[i, 0].set_title(f'Input Signal {i+1}')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Intensity')


        # Plot output signals
        for i in range(ports):
            axs[i, 1].plot(t, jnp.abs(outputscheck[f'o{i}'])**2)
            axs[i, 1].set_title(f'Output Signal o{i}')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('Intensity')
            
        plt.tight_layout()
        plt.show()






        

        # for i, model in enumerate(modelList):
        #     outputs,tr,x_out = model.response(inputs)
        #     ports = len(outputs)
        #     fig, axs = plt.subplots(ports, 2, figsize=(10, 10))
        #     for i in range(ports):
        #         axs[i, 0].plot(t, jnp.abs(inputs[f'o{i}'])**2)
        #         axs[i, 0].set_title(f'Input Signal {i+1}')
        #         axs[i, 0].set_xlabel('Time (s)')
        #         axs[i, 0].set_ylabel('Intensity')


        #     # Plot output signals
        #     for i in range(ports):
        #         axs[i, 1].plot(t, jnp.abs(outputs[f'o{i}'])**2)
        #         axs[i, 1].set_title(f'Output Signal o{i}')
        #         axs[i, 1].set_xlabel('Time (s)')
        #         axs[i, 1].set_ylabel('Intensity')


        #     # Adjust layout
        #     plt.tight_layout()
        #     plt.show()
        #     plt.plot(t,jnp.abs(x_out)**2)
        #     plt.show()
        

    general_function(instances, connections, ports, active_components)
    # general_function(instances, connections2, ports2, active_components)
        

