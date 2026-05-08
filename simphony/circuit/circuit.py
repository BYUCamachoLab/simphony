import inspect
import networkx as nx
from typing import cast
from collections.abc import Iterator
# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
from ipysigma import Sigma
from IPython.display import display
# import sax
# from jax.typing import ArrayLike
# from sax.saxtypes import Model as SaxModel

from simphony.circuit.netlist import add_settings_to_netlist, graph_to_netlist, netlist_to_graph, instantiated_flat_netlist_to_graph, remove_instances_from_netlist
from copy import deepcopy
import warnings
import inspect
# from simphony.signal import optical_signal, complete_steady_state_inputs

# from simphony.component.component import Component
# from simphony.component.pcell import PCell
from simphony.simulation.simulation import SimulationParameters


from simphony.libraries.ideal.s_parameters import optical_s_parameter_placeholder, SParameterPlaceholder, s_parameter_netlist_to_pcell
import sax

from sax.circuits import _create_dag
from sax.netlists import convert_nets_to_connections

from typing import Tuple
from simphony.libraries._internal.port_label import PortLabel, DirectedPortLabel, BidirectionalPortLabel
# import re


from simphony.libraries._internal.port_label import PortLabel, DirectedPortLabel, BidirectionalPortLabel
# from simphony.utils import dict_to_matrix

COMPONENT_COLOR_DEFAULT = "black"
COMPONENT_COLOR_SPARAM = "blue"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"
COMPONENT_COLOR_LOGIC = "gray"

S_PARAMETER_SETTING_KEYS = {
    "group_id",
    "vector_fitting_parameters",
    "delay_compensation",
    "port_directionality",
}

# -----------------------------------------------------------------
# The following functions taken from sax.circuits from sax 0.15.10
# -----------------------------------------------------------------
def _create_dag(
    netlist: sax.RecursiveNetlist,
    models: sax.Models | None = None,
    *,
    validate: bool = False,
) -> nx.DiGraph:
    if models is None:
        models = {}

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.items():
        if model_name not in all_models:
            all_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if component not in all_models:
                all_models[component] = models.get(component, None)
                g.add_node(component)
            g.add_edge(model_name, component)

    # we only need the nodes that depend on the parent...
    parent_node = next(iter(netlist.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = cast(nx.DiGraph, nx.induced_subgraph(g, nodes))
    if validate:
        g = _validate_dag(g)
    return g

def _validate_dag(dag: nx.DiGraph) -> nx.DiGraph:
    nodes = _find_root(dag)
    if len(nodes) > 1:
        msg = f"Multiple top_levels found in netlist: {nodes}"
        raise ValueError(msg)
    if len(nodes) < 1:
        msg = "Netlist does not contain any nodes."
        raise ValueError(msg)
    if not dag.is_directed():
        msg = "Netlist dependency cycles detected!"
        raise ValueError(msg)
    return dag

def _in_degree(dag: nx.DiGraph) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.in_degree())


def _out_degree(dag: nx.DiGraph) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.out_degree())

def _find_root(g: nx.DiGraph) -> list[str]:
    return [n for n, d in _in_degree(g) if d == 0]


def _find_leaves(g: nx.DiGraph) -> list[str]:
    return [n for n, d in _out_degree(g) if d == 0]

# -----------------------------------------------------------------
# End of functions taken from sax.circuits from sax 0.15.10
# -----------------------------------------------------------------

class Circuit:
    def __init__(
        self,
        netlist: dict,
        models: dict,
    ) -> None:
        # Validate the netlist
        # for key in netlist['instances'].keys():
        #     if not key.isidentifier():
        #         raise ValueError("All instance names must be valid python identifiers")

        
        self.netlist = sax.netlist(deepcopy(netlist))
        
        # Sanitize the names to make netlist valid
        new_netlist = convert_nets_to_connections(self.netlist) # necessary for gdsfactory netlists
        if not new_netlist == {}:
            self.netlist = new_netlist
        # Replace the original names
        
        for _, subnetlist in self.netlist.items():
            add_settings_to_netlist(subnetlist)

        self.netlist = sax.netlist(self.netlist)
        # self.flattened_netlist = sax.flatten_netlist(self.netlist)
        self.subcircuit_hierarchy = _create_dag(self.netlist)
        
        self.models = deepcopy(models)
        self._convert_sax_models()
        for instance_name, component in self.models.items():
            # component._create_port_lookup_tables()
            component._create_port_lookup_table()

        # self.flattened_graph = netlist_to_graph(self.flattened_netlist)
        pass
        # self._add_ports_to_graph(self.flattened_graph)
        # self._validate_connections(self.flattened_graph)

    def add_component(
        self,
        instance_name,
        model_name,
        model=None,
    ):
        """
        Add a component instance to the top-level netlist.

        Parameters
        ----------
        instance_name : str
            Name of the instance in the netlist.

        model_name : str
            Name of the component model.

        model : optional
            Optional model object to insert into self.models.
        """

        root = _find_root(self.subcircuit_hierarchy)[0]

        if instance_name in self.netlist[root]["instances"]:
            raise ValueError(f"Instance '{instance_name}' already exists")

        # Add model if provided
        if model is not None:
            self.models[model_name] = model

            if not inspect.isclass(self.models[model_name]):
                s_parameter = optical_s_parameter_placeholder(
                    self.models[model_name]
                )
                self.models[model_name] = s_parameter

            self.models[model_name]._create_port_lookup_table()

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        self.netlist[root]["instances"][instance_name] = {
            "component": model_name,
            "settings": {},
        }

        return self

    def add_connection(
        self,
        src_instance,
        src_port,
        dst_instance,
        dst_port,
    ):
        """
        Add a connection between two component ports.
        """

        root = _find_root(self.subcircuit_hierarchy)[0]

        if "connections" not in self.netlist[root]:
            self.netlist[root]["connections"] = {}

        src = f"{src_instance},{src_port}"
        dst = f"{dst_instance},{dst_port}"

        # SAX-style directed connection representation
        self.netlist[root]["connections"][src] = dst

        return self

    # def add_connection(
    #     self,
    #     src_instance,
    #     src_port,
    #     dst_instance,
    #     dst_port,
    # ):
    #     """
    #     Add a connection between two component ports.
    #     """

    #     root = _find_root(self.subcircuit_hierarchy)[0]

    #     connection_name = (
    #         f"{src_instance},{src_port}:{dst_instance},{dst_port}"
    #     )

    #     if "connections" not in self.netlist[root]:
    #         self.netlist[root]["connections"] = {}

    #     self.netlist[root]["connections"][connection_name] = (
    #         f"{src_instance},{src_port}",
    #         f"{dst_instance},{dst_port}",
    #     )

    #     return self


    # def unconnected_ports(
    #     self,
    #     inputs_only: bool = False,
    # ):
    #     """
    #     Return all unconnected ports as

    #     {
    #         instance_name: [port_name1, port_name2, ...]
    #     }

    #     Parameters
    #     ----------
    #     inputs_only : bool
    #         If True, only return ports whose directionality is
    #         "input" or "bidirectional".
    #     """

    #     root = _find_root(self.subcircuit_hierarchy)[0]
    #     netlist = self.netlist[root]

    #     # Start with all relevant ports assumed unconnected
    #     unconnected_ports = {}

    #     for instance_name, instance_data in netlist["instances"].items():

    #         model_name = instance_data["component"]

    #         if model_name not in self.models:
    #             continue

    #         component = self.models[model_name]

    #         if inputs_only:
    #             port_names = [
    #                 port.name
    #                 for port in component.ports
    #                 if port.directionality in ("input", "bidirectional")
    #             ]
    #         else:
    #             port_names = [
    #                 port.name
    #                 for port in component.ports
    #             ]

    #         unconnected_ports[instance_name] = set(port_names)

    #     # Remove connected ports
    #     for src, dst in netlist.get("connections", {}).items():

    #         for endpoint in (src, dst):
    #             instance_name, port_name = endpoint.split(",")

    #             if instance_name in unconnected_ports:
    #                 unconnected_ports[instance_name].discard(port_name)

    #     # Convert sets to sorted lists
    #     unconnected_ports = {
    #         instance: sorted(list(ports))
    #         for instance, ports in unconnected_ports.items()
    #         if len(ports) > 0
    #     }

    #     return unconnected_ports
    def unconnected_ports(
        self,
        inputs_only: bool = False,
    ):
        """
        Return all unconnected ports as

        {
            instance_name: [port_obj1, port_obj2, ...]
        }

        Parameters
        ----------
        inputs_only : bool
            If True, only return ports whose directionality is
            "input" or "bidirectional".
        """

        root = _find_root(self.subcircuit_hierarchy)[0]
        netlist = self.netlist[root]

        # instance -> set(port objects)
        unconnected_ports = {}

        for instance_name, instance_data in netlist["instances"].items():

            model_name = instance_data["component"]

            if model_name not in self.models:
                continue

            component = self.models[model_name]

            if inputs_only:
                ports = [
                    port
                    for port in component.ports
                    if port.directionality in ("input", "bidirectional")
                ]
            else:
                ports = list(component.ports)

            unconnected_ports[instance_name] = set(ports)

        # Remove connected ports (match by name)
        for src, dst in netlist.get("connections", {}).items():

            for endpoint in (src, dst):
                instance_name, port_name = endpoint.split(",")

                if instance_name in unconnected_ports:

                    # remove the matching port object
                    unconnected_ports[instance_name] = {
                        p for p in unconnected_ports[instance_name]
                        if p.name != port_name
                    }

        # Convert sets to lists
        unconnected_ports = {
            instance: list(ports)
            for instance, ports in unconnected_ports.items()
            if len(ports) > 0
        }

        return unconnected_ports

    # def add_component(
    #     self,
    #     instance_name,
    #     model_name,
    #     model = None,
    # ):
    #     # Don't forget to update the port_lookup_table, you might as well just call _create_port_lookup_table
    #     pass

    # def add_connection(
    #     self,
    #     instance_name,
    #     model_name,
    #     model = None,
    # ):
    #     # Don't forget to update the port_lookup_table, you might as well just call _create_port_lookup_table
    #     pass

    # def unconnected_ports(
    #     self,
    # ):
    #     # Look through all of the models in the models dict, simphony models are different than 
    #     # Sax models in that they have a ports attribute self.models[model_name].ports
    #     # Each port in self.models[model_name].ports has a port.name attribute
    #     # Alternatively, there is a port
    #     unconnected_ports = ... 
    #     return unconnected_ports

    def display(
        self, 
        subcircuit: str = None,
        inline: bool = True,
        node_labels: dict = {},
    ):
        """
        The true instance name is often not desirable for a node label.
        When an abbreviated or modified instance name is required, it 
        can be specified in the "node_labels" field.

        Any instance name that is not a key in the dict, will be used
        as the default node label.        
        """
        recursive_netlist = {}

        if subcircuit is None:
            subcircuit = _find_root(self.subcircuit_hierarchy)[0]
        
        recursive_netlist = self.get_subnetlist(subcircuit)

        # if flatten:
        #     netlist = sax.flatten_netlist(recursive_netlist)
        #     graph = netlist_to_graph(netlist)
        #     self._mark_component_types(subcircuit, graph)
        #     self._color_nodes(graph)
        # else:
        netlist = recursive_netlist[subcircuit]
        graph = netlist_to_graph(netlist, self.models, include_ports=True)
        # add_ports_to_graph(graph, netlist, self.models)
        # self._mark_component_types(subcircuit, graph)
        # self._color_nodes(graph)

        # # self._add_data_to_graph(graph)
        
        relabeled_graph = nx.relabel_nodes(graph, node_labels)

        # relabeled_graph.add_node(
        #     f"Kablooey",
        #     # component=instance["component"],
        #     # settings=instance["settings"],
        #     shape="rectangle",
        #     opacity=1.0,
        #     border_color="blue",
        #     color="white",
        #     size=20,
        #     border_size=1,
        #     # image="image.png",
        #     # opacity=0.5,
        #     # size=5,
        # )

        safe_graph = _sanitize_graph_for_widget(relabeled_graph)
        fig = Sigma(safe_graph, node_size=safe_graph.degree, node_color="club", start_layout = True)
        display(fig)
    
    def flatten(
        self,
        separator = "~",
    ):
        return FlatCircuit(self.netlist, self.models, separator=separator)
    
    def instantiate(
        self,
        settings: dict,
        simulation_parameters: SimulationParameters,
        tracked_ports: dict = None,
        directed: bool = False,
        fuse_models: bool = False,
        # default_modes,
    ):
        return InstantiatedCircuit(
            self, 
            settings,
            simulation_parameters,
            tracked_ports=tracked_ports,
            directed=directed,
            fuse_models=fuse_models
            # directed, 
            # default_modes
        )

    def get_subnetlist(self, subcircuit: str):
        original_netlist = deepcopy(self.netlist)
        if not subcircuit in self.subcircuit_hierarchy.nodes:
                return ValueError(f"{subcircuit} not in circuit. Did you mean {list(self.subcircuit_hierarchy.nodes)}?")
        descendants = nx.descendants(self.subcircuit_hierarchy, subcircuit)
        subcircuits_to_keep = set([subcircuit] + list(descendants)) - set(_find_leaves(self.subcircuit_hierarchy))
        
        return sax.netlist({key: original_netlist[key] for key in subcircuits_to_keep})


    def _convert_sax_models(self):
        for model in self.models:
            component = self.models[model]
            if not inspect.isclass(component):
                s_parameter = optical_s_parameter_placeholder(component)
                self.models[model] = s_parameter

    def _get_ports_from_subcircuit(self, subcircuit: str):
        # First, get a flat netlist of just the subcircuit
        recursive_netlist = self.get_subnetlist(subcircuit)
        flat_netlist = sax.flatten_netlist(recursive_netlist)
        ports = []
        for subcircuit_port_name, component_port_attr in flat_netlist['ports'].items():
            instance_name, instance_port_name = tuple(component_port_attr.split(','))
            component_name = flat_netlist['instances'][instance_name]['component']
            port = self.models[component_name]._port_lookup_table[instance_port_name]
            ports.append(port)
        
        return ports

    def _mark_component_types(self, subcircuit, graph):
        """ 
        """
        for instance, attr in graph.nodes.items():
            
            component_name = self.netlist[subcircuit]['instances'][instance]['component']
            if component_name in self.models:
                component_name = self.netlist[subcircuit]['instances'][instance]['component']
                ports = self.models[component_name].ports
            elif component_name in self.subcircuit_hierarchy.nodes:
                ports = self._get_ports_from_subcircuit(component_name)

            tags = set()
            for port in ports:
                if port.type == "electrical":
                    tags.add("electrical")
                elif port.type == "optical":
                    tags.add("optical")
                elif port.type == "logic":
                    tags.add("logic")

            graph.nodes[instance]["type"] = "/".join(sorted(tags))

    def _add_ports_to_graph(self, graph):
        for subcircuit, attr in graph.nodes.items():
            graph.nodes[subcircuit]["electrical ports"] = []
            graph.nodes[subcircuit]["logic ports"] = []
            graph.nodes[subcircuit]["optical ports"] = []

            model = attr["component"]
            component = self.models[model]
            # component_name = flat_netlist['instances'][instance_name]['component']

            electrical_ports = []
            optical_ports = []
            logic_ports = []

            for port in component.ports:
                if port.type == "electrical":
                    electrical_ports.append(port.name)
                elif port.type == "optical":
                    optical_ports.append(port.name)
                elif port.type == "logic":
                    logic_ports.append(port.name)
            
            graph.nodes[subcircuit]["electrical ports"] = electrical_ports
            graph.nodes[subcircuit]["optical ports"] = optical_ports
            graph.nodes[subcircuit]["logic ports"] = logic_ports

    def get_port_type(self, graph, instance, port):
        optical_ports = graph.nodes[instance]["optical ports"]
        electrical_ports = graph.nodes[instance]["electrical ports"]
        logic_ports = graph.nodes[instance]["logic ports"]

        if port in optical_ports:
            return "optical"
        elif port in electrical_ports:
            return "electrical"
        elif port in logic_ports:
            return "logic"

    def _validate_connections(self, graph):
        # Verify optical-to-optical, electrical-to-electrical, logic-to-logic
        for edge in graph.edges:
            src, dst, _ = edge
            
            src_port = graph.edges[edge]["src_port"]
            src_port_type = self.get_port_type(graph, src, src_port)

            dst_port = graph.edges[edge]["dst_port"]
            dst_port_type = self.get_port_type(graph, dst, dst_port)

            if not src_port_type == dst_port_type:
                raise ValueError("Port types must match")
        
        # TODO: Verify out to in or out to bidirectional connections

    def _color_nodes(self, graph):
        color = COMPONENT_COLOR_DEFAULT

        # Assumes tags in alphabetical order
        for instance in graph.nodes:
            if graph.nodes[instance]["type"] == "s-parameter":
                color = COMPONENT_COLOR_SPARAM
            elif graph.nodes[instance]["type"] == "electrical/optical":
                color = COMPONENT_COLOR_OPTOELECTRICAL
            elif graph.nodes[instance]["type"] == "optical":
                color = COMPONENT_COLOR_OPTICAL
            elif graph.nodes[instance]["type"] == "electrical":
                color = COMPONENT_COLOR_ELECTRICAL
            elif graph.nodes[instance]["type"] == "logic":
                color = COMPONENT_COLOR_LOGIC
            elif graph.nodes[instance]["type"] == "electrical/logic":
                color = COMPONENT_COLOR_ELECTRICAL
            elif graph.nodes[instance]["type"] == "logic/optical":
                color = COMPONENT_COLOR_ELECTRICAL
            elif graph.nodes[instance]["type"] == "electrical/logic/optical":
                color = COMPONENT_COLOR_OPTOELECTRICAL

            graph.nodes[instance]["color"] = color

class FlatCircuit:
    def __init__(
        self,
        netlist: dict,
        models: dict,
        separator: str = "~"
    ) -> None:
        # To avoid rewriting code, FlatCircuit mostly a wrapper for Circuit
        self.separator = "~"
        self._recursive_circuit = Circuit(netlist, models)
        self.netlist = sax.flatten_netlist(self._recursive_circuit.netlist, sep=separator)
        self._sanitized_netlist, self._sanitized_netlist_lut = self._sanitize_netlist(self.netlist, separator)
        self._circuit = Circuit(self._sanitized_netlist, models)
        self.models = self._circuit.models


    def display(
        self, 
        inline: bool = True,
        node_labels: dict = {},
    ):
        
        sanitized_node_labels = {v:k for k, v in self._sanitized_netlist_lut.items()}
        sanitized_node_labels_to_modify = {self._sanitized_netlist_lut[k]:v for k, v in node_labels.items()}
        for k, v in sanitized_node_labels_to_modify.items():
            sanitized_node_labels[k] = v
        
        self._circuit.display(node_labels=sanitized_node_labels)
    
    def instantiate(
        self,
        settings: dict,
        simulation_parameters: SimulationParameters,
        directed: bool = False,
        fuse_models: bool = False,
    ):
        return InstantiatedCircuit(
            self, 
            settings,
            simulation_parameters, 
            directed=directed,
            fuse_models=fuse_models
        )

    def _sanitize_netlist(
        self, 
        netlist, 
        separator
    ) -> Tuple[dict, dict]:
        """
        This function replaces all non-valid characters (characters not allowed
        in a valid python identifier, i.e. "~", "|", et cetera) in a flattened
        netlist with a valid, unique substring.

        Returns the sanitized flat netlist dict and 
        
        Since sax recursive netlists require instance names to be
        valid python identifiers, and since the separator argument
        in sax.flatten_netlist needs to be a non-valid character in 
        a python identifier in order to preserve uniqueness. We need 
        to be able to replace the separator, with a substring not found
        in any of the instance names before passing to the Circuit class 
        __init__ funciton, which assumes all instance names are valid
        python identifiers.
        
        The method for creating a unique separator is simple:
        1. Determine the largest number, M, of consecutive "_"'s in the instances
        2. Prepend and append M "_"'s to the string "SEPARATOR"
        """
        unique_str = "_"
        for key in netlist['instances'].keys():
            while unique_str in key:
                unique_str += "_"
        new_separator = unique_str + "SEPARATOR" + unique_str
        lut = {}
        for key in netlist['instances'].keys():
            lut[key] = key.replace(separator, new_separator)
        
        sanitized_netlist = {
            "instances": {},
            "connections": {},
            "ports": {},
        }
        
        for instance_name, instance_data in netlist["instances"].items():
            sanitized_instance_name = lut[instance_name]
            sanitized_netlist['instances'][sanitized_instance_name] = instance_data
        
        for src, dst in netlist["connections"].items():
            src_name, src_port = src.split(",")
            dst_name, dst_port = dst.split(",")
            
            sanitized_src = lut[src_name] + "," + src_port
            sanitized_dst = lut[dst_name] + "," + dst_port

            sanitized_netlist['connections'][sanitized_src] = sanitized_dst

        for external_port, internal_port_data in netlist['ports'].items():
            instance_name, internal_port = internal_port_data.split(",")
            sanitized_netlist['ports'][external_port] = lut[instance_name] + "," + internal_port
        
        return sanitized_netlist, lut

    # def _determine_block_mode_order(self):

# TODO: Put these functions in the instantiated circuit class as methods
# def _recover_s_parameter_placeholder_settings(
#     instantiated_circuit,
#     settings,
# ):
#     ...

def _add_directionality_settings_to_s_parameter_placeholders(
    # self,
    instantiated_circuit,
    settings,
    directed = False,
):
    """
    Important: Currently, simphony assumes all ports with an ambiguous directionality are source nodes (input) if the corresponding instance 
    appears in the values of the connection dict in the netlist. Otherwise (if the port is unconnected or in the keys of the connections dict)
    we label the port as "output". This convention is used assuming elsewhere, for example, the _insert_port_labels function depends on this
    assumption.
    """
    netlist = instantiated_circuit.instantiated_flat_netlist
    # models = flat_circuit.models
    for instance_name, instance_data in netlist['instances'].items():
        # component_class = models[instance_data['component']]
        component = netlist['instances'][instance_name]['model']
        if isinstance(component, SParameterPlaceholder):
            if not instance_name in settings:
                settings[instance_name] = component.settings
            # if not "sax_settings" in settings[instance_name].keys():
            #     settings[instance_name] = {"sax_settings": settings[instance_name]}
            # TODO: DOUBLE CHECK DEFAULT DICTIONARY CONSTRUCTION FOR EDGE CASES
            if directed:
                default_directionalities = {p.name:"output" if f"{instance_name},{p.name}" in netlist['connections'].keys() else "input" for p in component.ports}
                default_directionalities = {k:d if not (f"{instance_name},{k}" in netlist['connections'].keys() or not f"{instance_name},{k}" in netlist['connections'].values()) else "output" for k,d in default_directionalities.items()}
            else:
                default_directionalities = {p.name:"bidirectional" for p in component.ports}                
            
            settings[instance_name]["port_directionality"] = default_directionalities | settings[instance_name].get("port_directionality", {})

# def find_clipped_edges(full_graph, subgraph_nodes):
#     subgraph_nodes = set(subgraph_nodes)
#     clipped = []

#     for u, v, key, data in full_graph.edges(keys=True, data=True):
#         if (u in subgraph_nodes) != (v in subgraph_nodes):
#             clipped.append((u, v, key, data))

#     return clipped

def find_clipped_edges(full_graph, subgraph_nodes):
    subgraph_nodes = set(subgraph_nodes)
    clipped = []

    for u, v, key, data in full_graph.edges(keys=True, data=True):
        if (u in subgraph_nodes) != (v in subgraph_nodes):
            clipped.append((u, v, key, data))

    # Remove bidirectional duplicates
    unique_clipped = []
    seen = set()

    for u, v, key, data in clipped:
        normalized = frozenset({
            (u, data["src_port"]),
            (v, data["dst_port"]),
        })

        if normalized not in seen:
            seen.add(normalized)
            unique_clipped.append((u, v, key, data))

    return unique_clipped

# def _normalize_settings(model, instance_name, settings):
#     original_model = model._sax_model
#     model_params = inspect.signature(original_model).parameters
#     instance_settings = settings[instance_name]
#     if any(setting in model_params for setting in instance_settings):
#         settings[instance_name] = {"sax_settings": instance_settings}
#     else:
#         settings[instance_name].setdefault("sax_settings", {})
#         warnings.warn(f"Sax settings were not specified and inside settings do not match model function call for {instance_name}. Appending an empty sax_setting dictionary for that component.")

class InstantiatedCircuit:
    """
    Similar to the Circuit, but composed of the instantiated models, themselves, not Component classes

    Will always be flattened 
    1. no recursively defined netlists 
    2. PCells have been flattened into base components

    Ultimately, the core identity of this class is a wrapper around the FlatCircuit Class, 
    except that each instance in the netlist has an extra field "simphony_model", which 
    is an instantiated component object.

    Additional utility methods are added here for use in Simulator classes

    """
    def __init__(
        self,
        circuit: Circuit | FlatCircuit,
        settings: dict,
        simulation_parameters: SimulationParameters,
        # consolidate_s_parameter_components = True, # I have decided that this will just be the thing to do
        tracked_ports: dict = None,
        directed = False, # TODO: Implement bidirectional interpretation of ambiguous s parameter models
        fuse_models = False,
        # directed: bool,
        # default_modes,
    ):
        """
        When `directed` is True, unspecified directionalities of SParameterPlaceholder objects will determined 
        based on the order of connection in the netlist
        """
        if not isinstance(tracked_ports, dict):
            tracked_ports = {}
        for ext_port_name, port_designator in circuit.netlist["top_level"]['ports'].items():
            tracked_ports.setdefault(ext_port_name, port_designator)
        
        # TODO: I was mutating the inputs so I deep copied them. TODO: assess this for efficiency
        circuit = deepcopy(circuit)
        settings = deepcopy(settings)
        # simulation_parameters = deepcopy(simulation_parameters)
        simulation_parameters = deepcopy(simulation_parameters) 
        
        if isinstance(circuit, FlatCircuit):
            self.circuit = circuit
        elif isinstance(circuit, Circuit):
            self.circuit = circuit.flatten()

        netlist = self.circuit.netlist
        models = self.circuit.models

        from simphony.libraries.ideal.s_parameters import SParameterPlaceholder
        # Reinterpret Sax Settings to optical_s_parameter Component settings
        # for instance_name, instance_settings in settings.items():
        for instance_name in netlist['instances'].keys():
            model = models[netlist['instances'][instance_name]['component']]
            if issubclass(model, SParameterPlaceholder) and not "sax_settings" in settings[instance_name].keys():
                settings[instance_name] = {"sax_settings": settings[instance_name]}
        
        # gv.d3(netlist_to_graph(netlist, models)).display()
        tracked_ports = self._insert_port_label_placeholders(netlist, settings, models, tracked_ports) 
        # gv.d3(netlist_to_graph(netlist, models)).display()

        from simphony.circuit.netlist import instantiate_netlist
        self.instantiated_flat_netlist = instantiate_netlist(netlist, models, settings, simulation_parameters)
        self.graph = instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist, include_ports=False)
        self.simulation_parameters = simulation_parameters
        
        _add_directionality_settings_to_s_parameter_placeholders(self, settings, directed=directed)
        self.settings = settings


        # gv.d3(instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist)).display()
        self._consolidate_s_parameter_components(fuse_models)
        # gv.d3(instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist)).display()
        self.instantiated_flat_netlist, self.port_lookup_table = self._remove_port_labels(self.instantiated_flat_netlist, tracked_ports)

        self.graph = instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist, include_ports=False)
        
        

    # def display(self, inline=True):
        
    #     # graph.add_edge("lf1~mzi1~bot_mod", "lf1~mzi2~bot_mod", directed=True, color="red", hover="Hi!", tooltip="delay = 12 ps")
    #     # graph.add_edge("lf1~mzi2~bot_mod", "lf1~mzi1~bot_mod", directed=True, color="red", hover="Hi!", tooltip="delay = 12 ps")
    #     graph = instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist, include_ports=True)
    #     fig = gv.d3(graph, edge_hover_tooltip=True)


    #     fig.display(inline=True)

    #     # fig = gv.d3(self.graph.to_undirected())
    #     # fig.display(inline=True)
    def display(self, inline=True):
        graph = instantiated_flat_netlist_to_graph(
            self.instantiated_flat_netlist,
            include_ports=True
        )
        safe_graph = deepcopy(graph)
        for _, attr in safe_graph.nodes(data=True):
            if "settings" in attr:
                attr["settings"] = stringify_dict_values(attr["settings"])

        safe_graph = _sanitize_graph_for_widget(safe_graph)
        fig = Sigma(safe_graph, node_size=safe_graph.degree, node_color="club")
        display(fig)

    def _consolidate_s_parameter_components(self, fuse_models):
        """
        Because Simphony Circuits will likely be composed of mostly sax-s-parameter elements,
        we include this extra functionality to optimize the placement of s-parameter models.

        This method will find all of the strongly connected s-parameter components and pass each subnetlist
        into a PCell factory. How the subnetlists are handled further will depend on how this subnetlist
        is interpreted by the PCell factory and how it handles different siulation modes
        """
        # TODO: Replace all of the sugraphs with SParameterGroupPlaceholder Components, to ease the stitching process
        # TODO: make a way to iterate over these new SParameterGroupPlaceholder instance names and subgraph

        s_parameter_only_graph = deepcopy(self.graph)
        for node in self.graph.nodes():
            model = self.instantiated_flat_netlist['instances'][node]['model']
            if not isinstance(model, SParameterPlaceholder):
                s_parameter_only_graph.remove_node(node)

        
        subgraphs = [s_parameter_only_graph.subgraph(c).copy() for c in nx.weakly_connected_components(s_parameter_only_graph)]
        TOP_LEVEL_NAME = "top_level"

        clipped_netlist = remove_instances_from_netlist(self.instantiated_flat_netlist, s_parameter_only_graph.nodes())
        instantiated_recursive_netlist = sax.netlist(clipped_netlist, top_level_name=TOP_LEVEL_NAME)

        for j, subgraph in enumerate(subgraphs):
            if j == 1:
                pass
            # The following line is likely leading to duplicate connections, since bidirectional connections
            # are represented with two different connections (since nx.multidigraph doesn't have "bidirectional connections")
            clipped_edges = find_clipped_edges(self.graph, subgraph)

            # ports = [src for (src, dst, key, data) in clipped_edges] # Use the clipped edges to determine the ports
            # TODO: IMPORTANT, when determining the ports, right now it only looks at clipped edges
            # This causes some important ports to be ignored leading to them disappearing in the pcell's ports
            # ports = {f"o{i}":f"{src},{data["src_port"]}" if src in subgraph else f"{dst},{data["dst_port"]}" for i, (src, dst, key, data) in enumerate(clipped_edges)}
            # subnetlist = graph_to_netlist(subgraph, port=ports)

            subnetlist = graph_to_netlist(subgraph)
            models = {subnetlist['instances'][node]['component']: self.instantiated_flat_netlist['instances'][node]['model'].sax_model for node in subgraph.nodes()}
            port_designators = set()
            for instance_name, instance_data in subnetlist['instances'].items():
                model_name = instance_data['component']
                # Changed this line on Friday 05-08-26
                sdict = sax.multimode(models[model_name](), self.simulation_parameters.mode_identifiers)
                single_mode_ports = sorted({
                    p.split("@")[0]
                    for edge in sdict.keys()
                    for p in edge
                })
                port_designators.update([f"{instance_name},{p}" for p in single_mode_ports])
            
            # external_port_designators = {f"{src},{data["src_port"]}" if src in subgraph else f"{dst},{data["dst_port"]}" for (src, dst, key, data) in clipped_edges}
            internal_port_designators = set()
            for src_designator, dst_designator in subnetlist['connections'].items():
                internal_port_designators.add(src_designator)
                internal_port_designators.add(dst_designator)
            
            external_port_designators = port_designators - internal_port_designators
            ports = {f"o{i}":ep_designator for i, ep_designator in enumerate(external_port_designators)}
            subnetlist['ports'] = ports
            port_directionality = {node: self.settings[node]['port_directionality'] for node in subgraph.nodes()}
            # default_modes = {node: self.instantiated_flat_netlist['instances'][node]['model'].default_modes for node in subgraph.nodes()}

            # _settings = {k: self.settings[k] for k in subnetlist['instances'].keys() if k in self.settings}

            # We need to add the settings determined by the s_parameter_placeholder objects
            _settings = {k: self.instantiated_flat_netlist['instances'][k]['model'].settings for k in subnetlist['instances'].keys()}


            s_parameter_pcell = s_parameter_netlist_to_pcell(subnetlist, models, port_directionality, self.simulation_parameters.mode_identifiers, fuse_models=fuse_models)
            
            # TODO: The following lines of code are repeated elsewhere and it would be good to wrap them up in a function
            instantiated_s_parameter_pcell = s_parameter_pcell(self.simulation_parameters, settings = _settings)
            pcell_netlist = instantiated_s_parameter_pcell._instantiated_netlist(self.simulation_parameters)
            
            s_parameter_group = f"sparameter_group{j}"
            instantiated_recursive_netlist[s_parameter_group] = pcell_netlist
            instantiated_recursive_netlist[TOP_LEVEL_NAME]['instances'][s_parameter_group] = {"component": s_parameter_group}
            


            port_lut = {v:k for k, v in ports.items()}
            # TODO: Make sure this isn't creating the duplicate connections that may be in clipped edges
            for (src, dst, key, data) in clipped_edges:
                src_port = data["src_port"]
                dst_port = data["dst_port"]
                if src in s_parameter_only_graph.nodes:
                    src_port = port_lut[f"{src},{src_port}"]
                    src = s_parameter_group
                elif dst in s_parameter_only_graph.nodes:
                    dst_port = port_lut[f"{dst},{dst_port}"]
                    dst = s_parameter_group
                
                src_connection = f"{src},{src_port}"
                dst_connection = f"{dst},{dst_port}" 
                instantiated_recursive_netlist[TOP_LEVEL_NAME]['connections'][src_connection] = dst_connection
                                        
        
        self.instantiated_flat_netlist = sax.flatten_netlist(instantiated_recursive_netlist)

    def _insert_port_label_placeholders(self, netlist, settings, models, tracked_ports):
        # First, we will take care of the tracked ports that are not connected to anything else
        new_tracked_ports = {}
        unconnected_tracked_ports = []
        for tracked_port_name, tracked_port_designator in tracked_ports.items():
            if not is_endpoint_connected(netlist, tracked_port_designator):
                unconnected_tracked_ports.append((tracked_port_name, tracked_port_designator))
        
        for tracked_port_name, tracked_port_designator in unconnected_tracked_ports:
            internal_instance_name, internal_port_name = tracked_port_designator.split(",")
            port_label_instance_name = f"{tracked_port_name}|PORT_LABEL"
            port_label_model_name = port_label_instance_name

            netlist["instances"][port_label_instance_name] = {'component': port_label_model_name, "settings": {}}
            settings[port_label_instance_name] = {"name": tracked_port_name, "designator": tracked_port_designator}
            model_name = netlist['instances'][internal_instance_name]['component']
            tracked_port = models[model_name]._port_lookup_table[internal_port_name]
            
            

            if tracked_port.directionality == "bidirectional":
                models[port_label_model_name] = BidirectionalPortLabel
            else:
                models[port_label_model_name] = DirectedPortLabel

            if tracked_port.directionality == "output":
                netlist['connections'][tracked_port_designator] = f"{port_label_instance_name},in"
                new_tracked_ports[tracked_port_name] = f"{port_label_instance_name},in"
            elif tracked_port.directionality == "bidirectional":
                netlist['connections'][tracked_port_designator] = f"{port_label_instance_name},port1"
                new_tracked_ports[tracked_port_name] = f"{port_label_instance_name},port1"
            elif tracked_port.directionality == "input":
                netlist['connections'][f"{port_label_instance_name},out"] = tracked_port_designator
                new_tracked_ports[tracked_port_name] = f"{port_label_instance_name},out"
        
        for key, _ in unconnected_tracked_ports:
            tracked_ports.pop(key)
        
        # Next, we will take care of the tracked ports that are further connected
        for tracked_ext_port_name, tracked_port_designator in tracked_ports.items():
            if tracked_ext_port_name == "gc_out":
                pass
            internal_instance_name, internal_port_name = tracked_port_designator.split(",")
            port_label_instance_name = f"{tracked_ext_port_name}|PORT_LABEL"
            port_label_model_name = port_label_instance_name
            # netlist["instances"][port_label_instance_name] = {'component': port_label_model_name, "settings": {"name": tracked_ext_port_name, "designator": tracked_port_designator}}
            netlist["instances"][port_label_instance_name] = {'component': port_label_model_name, "settings": {}}
            settings[port_label_instance_name] = {"name": tracked_ext_port_name, "designator": tracked_port_designator}
            model_name = netlist['instances'][internal_instance_name]['component']
            tracked_port = models[model_name]._port_lookup_table[internal_port_name]
            
            if tracked_port.directionality == "bidirectional":
                models[port_label_model_name] = BidirectionalPortLabel
            elif tracked_port.directionality == "input" or tracked_port.directionality == "output":
                models[port_label_model_name] = DirectedPortLabel

            reverse_order = False
            if tracked_port_designator in netlist['connections'].keys():
                the_other_port_designator = netlist['connections'][tracked_port_designator]
                del netlist['connections'][tracked_port_designator]
            elif tracked_port_designator in netlist['connections'].values():
                reverse_order = True
                flipped_connections = {v:k for k, v in netlist['connections'].items()} # I put this here since the connections are mutated by inserted placeholders
                k = flipped_connections[tracked_port_designator]
                the_other_port_designator = k
                del netlist['connections'][k]

            def insert_port_label(netlist, tracked_designator, other_designator, port_names=("in", "out"), reverse_order=False):
                if reverse_order:
                    netlist["connections"][f"{port_label_instance_name},{port_names[0]}"] = tracked_designator
                    netlist["connections"][other_designator] = f"{port_label_instance_name},{port_names[1]}"
                    new_tracked_ports[tracked_ext_port_name] = f"{port_label_instance_name},{port_names[0]}"

                else:
                    netlist["connections"][tracked_designator] = f"{port_label_instance_name},{port_names[0]}"
                    netlist["connections"][f"{port_label_instance_name},{port_names[1]}"] = other_designator
                    new_tracked_ports[tracked_ext_port_name] = f"{port_label_instance_name},{port_names[0]}"
            
            if tracked_port.directionality == "bidirectional":
                insert_port_label(netlist, tracked_port_designator, the_other_port_designator, port_names=("port1", "port2"), reverse_order=reverse_order)
            elif tracked_port.directionality == "output":
                insert_port_label(netlist, tracked_port_designator, the_other_port_designator, port_names=("in", "out"), reverse_order=reverse_order)
            elif tracked_port.directionality == "input":
                insert_port_label(netlist, tracked_port_designator, the_other_port_designator, port_names=("out", "in"), reverse_order=reverse_order)
        return new_tracked_ports
            
    # # TODO: Test this function
    # # TODO: Currently, if two adjacent tracked ports are specified, things will break.
    # def _insert_port_label_placeholders(self, netlist, settings, models, tracked_ports):
    #     """
    #     Because pcells are recursively expanded into an unpredictable (from the perspective of this class) networks of simphony components,
    #     we insert a placeholder component to "track" were the ports in the top level netlist "end up"

    #     If the tracked port is an input port, the resulting placeholder will point into the port. The external facing port in the placeholder will match the tracked port and will be an input port

    #     If the tracked port is an output port, the tracked port will point into the placeholder. The external facing port in the placeholder will match the tracked port and will be an output port

    #     If the tracked port is bidirectional, then the placeholder will be connected to the tracked port via a bidirectional port and the external facing port will also be bidirectional. 
    #     """
        
        
    #     for tracked_ext_port_name, tracked_port_designator in tracked_ports.items():
    #         # TODO: optimize the following line
    #         flipped_connections = {v:k for k, v in netlist['connections'].items()} # I have to do this, otherwise, I can't have adjacent labels
            
    #         internal_instance_name, internal_port_name = tracked_port_designator.split(",")
    #         port_label_instance_name = f"{tracked_ext_port_name}|PORT_LABEL"
    #         port_label_model_name = port_label_instance_name
    #         # netlist["instances"][port_label_instance_name] = {'component': port_label_model_name, "settings": {"name": tracked_ext_port_name, "designator": tracked_port_designator}}
    #         netlist["instances"][port_label_instance_name] = {'component': port_label_model_name, "settings": {}}
    #         settings[port_label_instance_name] = {"name": tracked_ext_port_name, "designator": tracked_port_designator}
    #         model_name = netlist['instances'][internal_instance_name]['component']
    #         tracked_port = models[model_name]._port_lookup_table[internal_port_name]
            
    #         if tracked_port.directionality == "bidirectional":
    #             models[port_label_model_name] = BidirectionalPortLabel
    #         elif tracked_port.directionality == "input" or tracked_port.directionality == "output":
    #             models[port_label_model_name] = DirectedPortLabel
            
    #         if tracked_port_designator in netlist['connections'].keys():
    #             the_other_port_designator = netlist["connections"][tracked_port_designator]
    #             del netlist['connections'][flipped_connections[the_other_port_designator]]
                
    #             if tracked_port.directionality == "output" or tracked_port.directionality == "bidirectional":
    #                 netlist["connections"][tracked_port_designator] = f"{port_label_instance_name},in"
    #                 netlist["connections"][f"{port_label_instance_name},out"] = the_other_port_designator
    #             elif tracked_port.directionality == "input":
    #                 # THESE TECHNICALLY VIOLATE ONE OF SIMPHONIES ASSUMPTIONS ABOUT PORT DIRECITONALITY
    #                 netlist["connections"][tracked_port_designator] = f"{port_label_instance_name},out"
    #                 netlist["connections"][f"{port_label_instance_name},in"] = the_other_port_designator

    #         elif tracked_port_designator in netlist['connections'].values():
    #             # Now, I need the key in the netlist that corresponds to the tracked_port_designator value
    #             # Is there an efficient and pythonic way to do this?
    #             the_other_port_designator = flipped_connections[tracked_port_designator]
    #             del netlist['connections'][flipped_connections[tracked_port_designator]]
    #             if tracked_port.directionality == "output" or tracked_port.directionality == "bidirectional":
    #                 netlist["connections"][f"{port_label_instance_name},in"] = tracked_port_designator
    #                 netlist["connections"][the_other_port_designator] = f"{port_label_instance_name},out"
    #             elif tracked_port.directionality == "input":
    #                 netlist["connections"][f"{port_label_instance_name},out"] = tracked_port_designator
    #                 netlist["connections"][the_other_port_designator] = f"{port_label_instance_name},in"
    #         else: # The tracked port is unconnected
    #             if tracked_port.directionality == "output" or tracked_port.directionality == "bidirectional":
    #                 netlist["connections"][tracked_port_designator] = f"{port_label_instance_name},in"
    #             elif tracked_port.directionality == "input":
    #                 netlist["connections"][f"{port_label_instance_name},out"] = tracked_port_designator


    def _remove_port_labels(self, netlist, tracked_ports):
        new_tracked_ports = {}
        port_labels = {instance_name:instance_data['model'].name for instance_name, instance_data in netlist['instances'].items() if isinstance(instance_data['model'], PortLabel)}
        external_port_labels = {}
        internal_port_labels = {}

        for instance_name, tracked_port_name in port_labels.items():
            endpoints = _find_port_label_endpoints(netlist, instance_name)
            if len(endpoints) == 1:
                external_port_labels[instance_name] = tracked_port_name
            elif len(endpoints) == 2:
                internal_port_labels[instance_name] = tracked_port_name
        
        for instance_name, tracked_port_name in external_port_labels.items():
            new_tracked_ports[tracked_port_name] = find_connected_endpoint(netlist, tracked_ports[tracked_port_name])

        netlist = remove_instances_from_netlist(netlist, external_port_labels.keys())

        for instance_name, tracked_port_name in internal_port_labels.items():
            endpoints = _find_port_label_endpoints(netlist, instance_name)
            if endpoints[0] in netlist['connections'].keys():
                src = endpoints[0]
                dst = endpoints[1]
            else:
                src = endpoints[1]
                dst = endpoints[0]
            new_tracked_ports[tracked_port_name] = find_connected_endpoint(netlist, tracked_ports[tracked_port_name])
            netlist = remove_instances_from_netlist(netlist, [instance_name])
            netlist['connections'][src] = dst
            pass


            # if len(endpoints) == 1 and endpoints[0] in netlist['connections'].keys():
            #     remove_instances_from_netlist(netlist, [instance_name])

            # elif len(endpoints) == 1 and endpoints[0] in netlist['connections'].values():
            #     remove_instances_from_netlist(netlist, [instance_name])

            # if endpoints[0] in netlist['connections'].keys():
            #     src = endpoints[0]
            #     dst = endpoints[1]
            # else:
            #     src = endpoints[1]
            #     dst = endpoints[0]

            # if len(endpoints) == 2:
            #     remove_instances_from_netlist(netlist, [instance_name])
        
        return netlist, new_tracked_ports

def _find_port_label_endpoints(netlist, instance_name):
    # if instance_name == 'lf2|PORT_LABEL':
    #     pass
    endpoints = []
    for src_designator, dst_designator in netlist['connections'].items():
        src_instance = src_designator.split(",")[0]
        dst_instance = dst_designator.split(",")[0]

        if instance_name == src_instance:
            endpoints.append(dst_designator)
        if instance_name == dst_instance:
            endpoints.append(src_designator)

    return endpoints

def stringify_dict_values(d):
    """Recursively convert all values in a dict to strings."""
    if not isinstance(d, dict):
        return str(d)

    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = stringify_dict_values(v)
        elif isinstance(v, (list, tuple, set)):
            out[k] = [stringify_dict_values(x) for x in v]
        else:
            out[k] = str(v)
    return out

def _sanitize_graph_for_widget(graph):
    """Convert graph attributes to widget-safe values for notebook renderers."""
    safe_graph = deepcopy(graph)
    for _, attr in safe_graph.nodes(data=True):
        for k, v in list(attr.items()):
            if isinstance(v, dict):
                attr[k] = stringify_dict_values(v)
            elif isinstance(v, (list, tuple, set)):
                attr[k] = [stringify_dict_values(x) for x in v]
            else:
                try:
                    # Keep simple JSON-like scalars as-is; stringify complex objects.
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        continue
                    attr[k] = str(v)
                except Exception:
                    attr[k] = str(v)
    for _, _, attr in safe_graph.edges(data=True):
        for k, v in list(attr.items()):
            if isinstance(v, (str, int, float, bool)) or v is None:
                continue
            attr[k] = str(v)
    return safe_graph

# import json

# def make_json_safe(obj):
#     """Convert object into something JSON serializable."""
#     # Fast path: already serializable
#     try:
#         json.dumps(obj)
#         return obj
#     except (TypeError, OverflowError):
#         pass

#     # Common conversions
#     if isinstance(obj, dict):
#         return {str(k): make_json_safe(v) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple, set)):
#         return [make_json_safe(v) for v in obj]
    
#     # JAX / NumPy arrays
#     try:
#         import numpy as np
#         if hasattr(obj, "shape"):
#             return np.array(obj).tolist()
#     except Exception:
#         pass

#     # Fallback: string representation
#     return str(obj)

# def sanitize_graph_for_display(graph):
#     import networkx as nx
#     G = nx.DiGraph()

#     # Copy nodes
#     for n, attrs in graph.nodes(data=True):
#         safe_attrs = {k: make_json_safe(v) for k, v in attrs.items()}
#         G.add_node(n, **safe_attrs)

#     # Copy edges
#     for u, v, attrs in graph.edges(data=True):
#         safe_attrs = {k: make_json_safe(v) for k, v in attrs.items()}
#         G.add_edge(u, v, **safe_attrs)

#     return G
def is_endpoint_connected(netlist, endpoint):
    """
    Check whether a SAX netlist endpoint is connected.

    Parameters
    ----------
    netlist : dict
        SAX-style netlist dictionary containing a "connections" dict.

    endpoint : str
        Endpoint in the form "instance_name,port_name"

    Returns
    -------
    bool
        False if the endpoint does not appear in any connection,
        True otherwise.
    """

    connections = netlist.get("connections", {})

    for src, dst in connections.items():
        if endpoint == src or endpoint == dst:
            return True

    return False

def find_bidirectional_duplicates(connections: dict):
    """
    Finds pairs (k, v) where both k->v and v->k exist.

    Returns:
        set of frozensets, each representing a duplicate pair
        (so {A, B} represents A<->B)
    """
    seen = set()
    duplicates = set()

    for k, v in connections.items():
        pair = (k, v)

        # normalize direction-independent representation
        reversed_pair = (v, k)

        if reversed_pair in seen:
            duplicates.add(frozenset(pair))
        else:
            seen.add(pair)

    return duplicates

def find_connected_endpoint(netlist, endpoint):
    """
    Find the endpoint connected to `endpoint` in a SAX netlist.

    Parameters
    ----------
    netlist : dict
        SAX-style netlist dictionary containing a "connections" dict.

    endpoint : str
        Endpoint in the form "instance_name,port_name"

    Returns
    -------
    str | None
        The connected endpoint if found, otherwise None.
    """

    connections = netlist.get("connections", {})

    for src, dst in connections.items():
        if endpoint == src:
            return dst
        elif endpoint == dst:
            return src

    return None