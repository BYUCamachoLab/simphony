import numpy as np
import matplotlib.pyplot as plt
import sax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix,pole_residue_to_time_system
from simphony.time_domain.utils import gaussian_pulse
from simphony.time_domain.pole_residue_model import IIRModelBaseband
from simphony.time_domain.utils import pole_residue_to_time_system
from simphony.libraries import siepic


class TimeDomainSim:
    def __init__(self, netlist:dict, models:dict):
        self.netlist = netlist
        self.models = models
        self.instances = {}

        for instance_name, model_name in self.netlist['instances'].items():
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' is not defined in self.models.")
            self.instances[instance_name] = self.models[model_name]

        used_model_names = set(self.netlist['instances'].values())
        all_model_names = set(self.models.keys())
        unused_model_names = all_model_names - used_model_names

        if unused_model_names:
            print(f"Warning: The following models are never called in the netlist: "
                  f"{unused_model_names}")
            
        self.connections = {instance: {} for instance in self.instances.keys()}
        for designation_a, designation_b in self.netlist['connections'].items():
            instance_a, port_a = map(str.strip, designation_a.split(','))
            instance_b, port_b = map(str.strip, designation_b.split(','))
            
            self.connections[instance_a][port_a] = (instance_b, port_b)
            self.connections[instance_b][port_b] = (instance_a, port_a)
        
        self.ports = {}
        for circuit_port, designation in self.netlist['ports'].items():
            instance_name, instance_port = map(str.strip, designation.split(','))
            self.ports[circuit_port] = (instance_name, instance_port)
            
        
            
    def build_model(self, 
                    wvl = np.linspace(1.5, 1.6, 200),
                    center_wvl = 1.55,
                    model_order = 50,
                    num_measurements = 200,
                    models = None,
                    model_parameter_list=None,
                    ):
        self.active = {}
        self.active_components = {}
        if 'active_components' in self.netlist:
            for active_name in self.netlist['active_components'].items():
                self.active_components[active_name] = active_name
            self.sub_netlists, self.removed_connections, self.removed_ports = self.create_passive_sub_netlists(self.instances, self.connections, self.ports, self.active_components)
            
            for i, nl in enumerate(self.sub_netlists):
                print(f"--- Sub-Netlist {i} ---")
                print("instances:", nl["instances"])
                print("connections:", nl["connections"])
                print("ports:", nl["ports"])
                print()

            sub_circuit_list = {}
            port_list = {}
            self.model_list = []
            for i, netlist in enumerate(self.sub_netlists):
                dt = 1e-12
                circuittemp, info = sax.circuit(
                    netlist=netlist,
                    models=models,
                )
                sub_circuit_list[i] = circuittemp
                port_list[i] = netlist['ports']
            for i, circuit in sub_circuit_list.items():
                circuit_params = self.create_circuit_parameters(
                    wl=wvl,
                    model_list_parameters=model_parameter_list,
                    components=models
                )
        
                # Execute circuit with generated parameters
                s = circuit(**circuit_params)
                
                # Continue with existing processing
                S = np.asarray(dict_to_matrix(s))
                self.modelList.append(IIRModelBaseband(wvl, center_wvl, S, model_order))



        else:
            circuit, info = sax.circuit(netlist = self.netlist, models= self.models)
            s = circuit(wl = wvl)
            self.S = np.asarray(dict_to_matrix(s))
            model = IIRModelBaseband(wvl,center_wvl,self.S, model_order)
            self.time_system = pole_residue_to_time_system(model)


    def create_circuit_parameters(self, wl, model_list_parameters, components):
        """Create SAX-compatible parameters from structured list input."""
        params = {'wl': wl}
        
        if model_list_parameters:
            for component_spec in model_list_parameters:
                comp_type = component_spec['type']
                suffix = component_spec.get('suffix', '')
                comp_params = component_spec.get('params', {})
                
                # Construct component name
                comp_name = comp_type + str(suffix)
                
                # Verify component exists
                if comp_type not in components:
                    raise ValueError(f"Component {comp_type} not found in model library")
                    
                # Add parameters
                params[comp_name] = {
                    **comp_params,
                    'model': components[comp_type]
                }
        
        return params




    def remove_active_edges_and_track_them(self,connections, active_components):
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


    def remove_ports_to_active(self,ports, active_components):
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


    def build_component_graph(self, connections, ports, active_components, directed=False):
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


    def tarjan_scc(self, graph):
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


    def build_sub_netlist(self,
        scc_components,
        all_connections,
        original_ports,
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
        
        # 3) Ports (original top-level) - keep only if they reference a component in the SCC
        scc_ports = {}
        for port_label, comp_port_str in original_ports.items():
            comp, port = comp_port_str.split(',')
            if comp in scc_components:
                scc_ports[port_label] = comp_port_str
        
        # 4) Add new external ports for each (passiveComp, activeComp) removed edge
        #    if passiveComp is in scc_components
        new_port_index = 0
        for (passive_comp, port_name) in removed_edges:
            if passive_comp in scc_components:
                
                new_label = f"active_boundary_{new_port_index}"
                new_port_index += 1
                # You could store the actual original port if you wanted, but we've lost that info
                # in this component-level approach. We'll just call it "portX".
                # For clarity, let's call it "portX" or "portA"
                scc_ports[new_label] = f"{passive_comp},{port_name}"
        
        new_netlist = {
            "instances": scc_instances,
            "connections": scc_connections,
            "ports": scc_ports,
        }
        return new_netlist


    def create_passive_sub_netlists(self,
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
        filtered_conns, removed_edges, removed_connecions = self.remove_active_edges_and_track_them(connections, active_components)
        
        # 2) Remove top-level ports referencing active comps
        filtered_ports, removed_ports = self.remove_ports_to_active(ports, active_components)
        
        # 3) Build a graph at component level (ignore port detail)
        graph = self.build_component_graph(filtered_conns, ports, active_components, directed=directed)
        
        # 4) Find SCCs
        sccs = self.tarjan_scc(self, graph)  # e.g. [ ['cr1','wg1','cr2'], ['cr3','wg3','cr4'], ... ]
        
        # 5) Build a netlist for each SCC
        sub_netlists = []
        for scc_comp_list in sccs:
            scc_comp_set = set(scc_comp_list)
            # Build the sub-netlist
            sub_nl = self.build_sub_netlist(
                scc_comp_set,
                filtered_conns,
                filtered_ports,
                removed_edges,
                instances
            )
            # If the SCC has at least one instance from 'instances', add it
            if sub_nl["instances"]:
                sub_netlists.append(sub_nl)
        
        return sub_netlists, removed_connecions, removed_ports
        
                
            