import inspect
import networkx as nx
from typing import cast
from collections.abc import Iterator
# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
import gravis as gv
# import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

from simphony.utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph
from copy import deepcopy
# from simphony.signal import optical_signal, complete_steady_state_inputs

import jax
import jax.numpy as jnp

from simphony.component.component import Component

from simphony.libraries.analytic.s_parameters import optical_s_parameter
import sax

from sax.circuits import _create_dag
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
        # default_settings: dict = None
    ) -> None:
        self.netlist = sax.netlist(deepcopy(netlist))
        
        # if 'instances' in netlist.keys():        
        for subnetlist_name, subnetlist in self.netlist.items():
            add_settings_to_netlist(subnetlist)

        self.recursive_netlist = sax.netlist(self.netlist)
        self.flattened_netlist = sax.flatten_netlist(self.recursive_netlist)
        self.subcircuit_hierarchy = _create_dag(self.recursive_netlist)
        
        self.models = models
        self._convert_sax_models()
        for instance_name, component in self.models.items():
            component._create_port_lookup_table()

        self.flattened_graph = netlist_to_graph(self.flattened_netlist)
        pass
        self._add_ports_to_graph(self.flattened_graph)
        self._validate_connections(self.flattened_graph)


    def display(
        self, 
        subcircuit: str = None, 
        flatten: bool = False, 
        inline: bool = True,
    ):
        """
        
        """
        recursive_netlist = {}
        # if subcircuit is not None:
        #     recursive_netlist = self.get_subnetlist(subcircuit)
        # else:
        #     recursive_netlist = deepcopy(self.recursive_netlist)

        if subcircuit is None:
            subcircuit = _find_root(self.subcircuit_hierarchy)[0]
        
        recursive_netlist = self.get_subnetlist(subcircuit)

        if flatten:
            netlist = sax.flatten_netlist(recursive_netlist)
            graph = netlist_to_graph(netlist)
            self._mark_component_types(subcircuit, graph)
            self._color_nodes(graph)
        else:
            netlist = recursive_netlist[subcircuit]
            graph = netlist_to_graph(netlist)
            self._mark_component_types(subcircuit, graph)
            self._color_nodes(graph)
            # self._add_data_to_graph(graph)
        
        fig = gv.d3(graph)
        fig.display(inline=inline)
    
    def get_subnetlist(self, subcircuit: str):
        original_netlist = deepcopy(self.recursive_netlist)
        if not subcircuit in self.subcircuit_hierarchy.nodes:
                return ValueError(f"{subcircuit} not in circuit. Did you mean {list(self.subcircuit_hierarchy.nodes)}?")
        descendants = nx.descendants(self.subcircuit_hierarchy, subcircuit)
        subcircuits_to_keep = set([subcircuit] + list(descendants)) - set(_find_leaves(self.subcircuit_hierarchy))
        # if key in original_netlist.keys()
        return sax.netlist({key: original_netlist[key] for key in subcircuits_to_keep})

    #Matthew's Suggestions
    #You don't remove ports with the component name attached. 
    #Is this on purpose with the understanding that these components don't have ports?
    #or is this a bug?

    def remove_components(self, components):
        components = list(components)
        self.graph.remove_nodes_from(components)
        
        # Remove from instances
        for component in components:
            self.netlist['instances'].pop(component, None)

        # Remove connections
        filtered_connections = {
            k: v for k, v in self.netlist['connections'].items()
            if not any(s in k or s in v for s in components)
        }
        self.netlist['connections'] = filtered_connections

        # Remove ports
        if 'ports' in self.netlist:
            filtered_ports = { 
                k: v for k, v in self.netlist['ports'].items()
                if not any(s in v for s in components)
            }
            self.netlist['ports'] = filtered_ports
        pass

    # def _add_data_to_flattened_graph(self, graph):
    #     self._mark_component_types(graph)
    #     # self._add_ports_to_graph(graph)
    #     # self._validate_connections(graph)
    #     self._color_nodes(graph)

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
            # component_name = self.flattened_netlist['instances'][subcircuit]
            # component_name = attr["component"]
            # component = self.models[component_name]
            # ports = component.ports

            if instance in self.subcircuit_hierarchy.nodes:
                ports = self._get_ports_from_subcircuit(instance)
            else:
                component_name = self.recursive_netlist[subcircuit]['instances'][instance]['component']
                ports = self.models[component_name].ports

            tags = set()
            for port in ports:
                if port.type == "electrical":
                    tags.add("electrical")
                elif port.type == "optical":
                    tags.add("optical")
                elif port.type == "logic":
                    tags.add("logic")

            # tags = set()
            # if component.electrical_port_names:
            #     tags.add("electrical")
            # if component.logic_port_names:
            #     tags.add("logic")
            # if component.optical_port_names:
            #     tags.add("optical")

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

            # if component.electrical_ports:
            #     self.graph.nodes[instance]["electrical ports"] = self.models[
            #         model
            #     ].electrical_ports
            # if component.logic_ports:
            #     self.graph.nodes[instance]["logic ports"] = self.models[
            #         model
            #     ].logic_ports
            # if component.optical_ports:
            #     self.graph.nodes[instance]["optical ports"] = self.models[
            #         model
            #     ].optical_ports

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
