import inspect
import networkx as nx
from typing import cast
from collections.abc import Iterator
# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
import gravis as gv
# import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

from simphony.circuit.netlist import add_settings_to_netlist, complete_netlist, get_settings_from_netlist, netlist_to_graph, instantiated_flat_netlist_to_graph
from copy import deepcopy
# from simphony.signal import optical_signal, complete_steady_state_inputs

import jax
import jax.numpy as jnp

from simphony.component.component import Component
from simphony.component.pcell import PCell
from simphony.simulation.simulation import SimulationParameters


from simphony.libraries.ideal.s_parameters import optical_s_parameter
import sax

from sax.circuits import _create_dag
from sax.netlists import convert_nets_to_connections

from typing import Tuple

import re
# from simphony.utils import dict_to_matrix

COMPONENT_COLOR_DEFAULT = "black"
COMPONENT_COLOR_SPARAM = "blue"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"
COMPONENT_COLOR_LOGIC = "gray"


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
        self.netlist = convert_nets_to_connections(self.netlist) # necessary for gdsfactory netlists
        # Replace the original names
        
        for _, subnetlist in self.netlist.items():
            add_settings_to_netlist(subnetlist)

        self.netlist = sax.netlist(self.netlist)
        # self.flattened_netlist = sax.flatten_netlist(self.netlist)
        self.subcircuit_hierarchy = _create_dag(self.netlist)
        
        self.models = models
        self._convert_sax_models()
        for instance_name, component in self.models.items():
            component._create_port_lookup_table()

        # self.flattened_graph = netlist_to_graph(self.flattened_netlist)
        pass
        # self._add_ports_to_graph(self.flattened_graph)
        # self._validate_connections(self.flattened_graph)


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
        graph = netlist_to_graph(netlist, self.models)
        self._mark_component_types(subcircuit, graph)
        self._color_nodes(graph)
        # self._add_data_to_graph(graph)
        
        relabeled_graph = nx.relabel_nodes(graph, node_labels)

        relabeled_graph.add_node(
            f"Kablooey",
            # component=instance["component"],
            # settings=instance["settings"],
            shape="rectangle",
            opacity=0.1,
            border_color="blue",
            color="white",
            size=20,
            border_size=1,
            # image="image.png",
            # opacity=0.5,
            # size=5,
        )

        fig = gv.d3(relabeled_graph.to_undirected(), edge_hover_tooltip=True)
        fig.display(inline=inline)
    
    def flatten(
        self,
        separator = "~",
    ):
        return FlatCircuit(self.netlist, self.models, separator=separator)
    
    def instantiate(
        self,
        simulation_parameters: SimulationParameters,
        settings: dict,
        # directed: bool,
        # default_modes,
    ):
        return InstantiatedCircuit(
            self, 
            simulation_parameters, 
            settings, 
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
                s_parameter = optical_s_parameter(component)
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
        settings,
        directed: bool,
        default_modes,
    ):
        return InstantiatedCircuit(self, settings, directed, default_modes)

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
        simulation_parameters: SimulationParameters,
        settings: dict,
        # directed: bool,
        # default_modes,
    ):
        if isinstance(circuit, FlatCircuit):
            self.circuit = circuit
        elif isinstance(circuit, Circuit):
            self.circuit = circuit.flatten()  

        netlist = self.circuit.netlist
        models = self.circuit.models

        from simphony.libraries.ideal.s_parameters import SParameterSax
        # Reinterpret Sax Settings to optical_s_parameter Component settings
        # for instance_name, instance_settings in settings.items():
        for instance_name in netlist['instances'].keys():
            model_name = netlist['instances'][instance_name]['component']
            if issubclass(models[model_name], SParameterSax) and not "sax_settings" in settings[instance_name].keys():
                settings[instance_name] = {"sax_settings": settings[instance_name]}
                pass
            pass

        from simphony.circuit.netlist import instantiate_netlist
        # Convert Sax Models First
        self.instantiated_flat_netlist = instantiate_netlist(netlist, models, settings, simulation_parameters)
        self.graph = instantiated_flat_netlist_to_graph(self.instantiated_flat_netlist)

    def display(self, inline=True):
        
        # graph.add_edge("lf1~mzi1~bot_mod", "lf1~mzi2~bot_mod", directed=True, color="red", hover="Hi!", tooltip="delay = 12 ps")
        # graph.add_edge("lf1~mzi2~bot_mod", "lf1~mzi1~bot_mod", directed=True, color="red", hover="Hi!", tooltip="delay = 12 ps")
        
        fig = gv.d3(self.graph, edge_hover_tooltip=True)


        fig.display(inline=True)

        # fig = gv.d3(self.graph.to_undirected())
        # fig.display(inline=True)

