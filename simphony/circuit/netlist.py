"""Netlist conversion and normalization helpers.

This module contains the lower-level utilities used by `Circuit` and the
simulators to move between SAX-style netlists, NetworkX graphs, and instantiated
flat netlists. Most users interact with these helpers indirectly through
`Circuit`, but they are useful when building examples, diagnostics, or custom
tooling around Simphony netlists.
"""

# from sax import AnyNetlist, InstanceName, Ports, Connections, Models
# import sax
# from typing import TypeAlias, TypedDict
# from simphony.component.component import Component
from simphony.component.pcell import PCell
# from typing_extensions import NotRequired
import networkx as nx
# import jax.numpy as jnp
import yaml
# from jax.scipy.special import factorial
from typing import Union
# from copy import deepcopy
from simphony.circuit._netlist import _instantiate_netlist, _add_settings_to_netlist, InstantiatedFlatNetlist
# from simphony.libraries.ideal.s_parameters import optical_s_parameter
from simphony.simulation.simulation import SimulationParameters
# from sax import DEFAULT_MODES
# import inspect

def instantiate_netlist(
    netlist: dict,
    models: dict,
    settings: dict,
    simulation_parameters: SimulationParameters,
    # directed: bool = False,
    # default_modes: tuple = DEFAULT_MODES,
)->InstantiatedFlatNetlist:
    """Instantiate a possibly hierarchical netlist.

    The helper delegates to the internal recursive instantiation routine. It
    expands PCells, applies per-instance settings, and returns a flat netlist
    whose instances contain concrete component objects under the `"model"` key.

    Parameters
    ----------
    netlist:
        SAX-style netlist, usually containing `instances`, `connections`, and
        `ports`.
    models:
        Mapping from component/model names to Simphony component classes, PCell
        classes, or converted SAX placeholders.
    settings:
        Per-instance constructor settings.
    simulation_parameters:
        Active simulation-mode parameters used when constructing components.

    Returns
    -------
    InstantiatedFlatNetlist
        Flattened netlist with instantiated component objects.
    """

    return _instantiate_netlist(netlist, models, settings, simulation_parameters)

    
    
    # for instance_name, instance_data in netlist['instances'].items():
    #     model = models[instance_data['component']]
    #     if issubclass(model, Component) or issubclass(model, PCell):
    #         continue
        
    #     if directed:
    #         directionality = ""
    #         model = optical_s_parameter(model, directionality, default_modes)
    #     else:
    #         directionality = "bidirectional"
    #         model = optical_s_parameter(model, directionality, default_modes)
    
    # return new_netlist, new_models


# def group_instances(netlist, group_name, instance_names):
#     new_netlist = deepcopy(netlist)

#     return new_netlist


def complete_netlist(netlist):
    """Ensure a flat netlist has the standard top-level keys.

    Missing `instances`, `connections`, or `ports` entries are inserted as empty
    dictionaries. The input dictionary is updated in place.

    """
    if 'instances' not in netlist:
        netlist['instances'] = {}
    if 'connections' not in netlist:
        netlist['connections'] = {}
    if 'ports' not in netlist:
        netlist['ports'] = {}


def add_settings_to_netlist(netlist, settings=None):
    """Attach settings dictionaries to each instance in a netlist.

    Parameters
    ----------
    netlist:
        Flat SAX-style netlist to update.
    settings:
        Optional mapping from instance name to settings. When omitted, each
        instance receives an empty settings dictionary.
    """
    return _add_settings_to_netlist(netlist, settings=settings)

def get_settings_from_netlist(netlist):
    """Extract per-instance settings from a flat netlist."""
    settings = {}
    for instance, attr in netlist['instances'].items():
        settings[instance] = attr['settings']
    
    return settings


def netlist_to_graph(netlist: Union[dict, str], models, include_ports=True):
    """Convert a flat netlist into a NetworkX multidigraph.

    Connection direction is derived from the declared `Port.directionality` of
    the two endpoints. Bidirectional-to-bidirectional connections are represented
    with edges in both directions, while directed combinations produce a single
    edge in the signal-flow direction.

    Parameters
    ----------
    netlist:
        Flat netlist dictionary or path to a YAML netlist file.
    models:
        Model library used to look up port metadata.
    include_ports:
        If true, add small graph nodes for top-level and unconnected ports so
        display tools can show the circuit boundary.
    """
    if isinstance(netlist, dict):
        pass
    elif isinstance(netlist, str):
        try:
            with open(netlist, "r") as file:
                netlist = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file '{netlist}' not found.")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    graph = nx.MultiDiGraph()

    for instance_name, instance in netlist["instances"].items():
        model = models[instance['component']]
        shape="rectangle"
        if issubclass(model, PCell):
            shape = "hexagon"

        graph.add_node(
            instance_name,
            component=instance["component"],
            settings=instance["settings"],
            size=25,
            color="white",
            border_color="black",
            border_size=1,
            shape=shape,
        )
        models[netlist['instances'][instance_name]['component']]._create_port_lookup_table()

    # Add edges based on connections
    for src, dsts in netlist["connections"].items():
        for dst in dsts.split(";"):
            if dst =='':
                continue
            src_instance, src_port = src.split(",")
            dst_instance, dst_port = dst.split(",")
            src_port_directionality = models[netlist['instances'][src_instance]['component']]._port_lookup_table[src_port].directionality
            dst_port_directionality = models[netlist['instances'][dst_instance]['component']]._port_lookup_table[dst_port].directionality
            print(f"dst: {dst}")
            print(f"dst_directionality: {dst_port_directionality}")
            add_connection_to_graph(graph, src_instance.strip(), dst_instance.strip(), src_port.strip(), dst_port.strip(), src_port_directionality, dst_port_directionality)
            # graph.add_edge(src_instance.strip(), dst_instance.strip(), src_port=src_port.strip(), dst_port=dst_port.strip())
    
    if include_ports:
        add_ports_to_graph(graph, netlist, models)

    #Matthew's Changes
    #Adds the ports as a graph attribute to be used within graph_to_netlist
    graph.graph["ports"] = netlist.get("ports", {}).copy()

    return graph

def add_ports_to_graph(graph, netlist, models):
    """Add top-level port nodes to a circuit graph.

    This helper is primarily used by `netlist_to_graph` for display graphs. It
    reads `netlist["ports"]`, looks up the connected internal port
    directionality, and adds boundary nodes connected to the owning component.
    """
    for external_port_name, internal_port_data in netlist["ports"].items():
        instance_name, internal_port_name = internal_port_data.split(",")
        node_name = f".{external_port_name}" # . Symbol Ensures Uniqueness of node_name
        directionality = models[netlist['instances'][instance_name]['component']]._port_lookup_table[internal_port_name].directionality
        add_port_to_graph(graph, instance_name, internal_port_name, directionality, external=True, external_port_name=external_port_name)

    unconnected_ports = set()
    connected_ports = {connection for connection in list(netlist['connections'].keys()) + list(netlist['connections'].values())}
    for instance_name, instance_data in netlist['instances'].items():
        for port in models[netlist['instances'][instance_name]['component']].ports:
            if not f"{instance_name},{port.name}" in connected_ports:
                unconnected_ports.add(f"{instance_name},{port.name}")

def add_connection_to_graph(graph, src_node, dst_node, src_port, dst_port, src_directionality, dst_directionality, port_type=None):
    """Add a connection edge using endpoint directionality rules.

    The source and destination in the netlist are treated as a physical
    connection description. The actual graph edge direction is chosen from the
    source/destination port directionality values. Bidirectional ports create
    reverse edges where needed.
    """
    if (src_directionality == "bidirectional" and dst_directionality == "bidirectional"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}↔{dst_node},{dst_port}", port_type=port_type)
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}↔{dst_node},{dst_port}", port_type=port_type)
    elif (src_directionality == "bidirectional" and dst_directionality == "input") or (src_directionality == "output" and dst_directionality == "bidirectional") or (src_directionality == "output" and dst_directionality == "input"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}→{dst_node},{dst_port}", port_type=port_type)
    elif (src_directionality == "bidirectional" and dst_directionality == "output") or (src_directionality == "input" and dst_directionality == "bidirectional") or (src_directionality == "input" and dst_directionality == "output"):
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}←{dst_node},{dst_port}", port_type=port_type)
    elif (src_directionality == "unknown" and dst_directionality== "input") or (src_directionality == "output" and dst_directionality== "unknown"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}→{dst_node},{dst_port}", port_type=port_type)
    elif (src_directionality == "unknown" and dst_directionality== "output") or (src_directionality == "input" and dst_directionality== "unknown"):
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}←{dst_node},{dst_port}", port_type=port_type)
    elif (src_directionality == "unknown" or dst_directionality== "unknown"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}?⎯?{dst_node},{dst_port}", color="red", type=port_type)
    else:
        raise ValueError(f"Cannot connect {src_directionality} to {dst_directionality}")

def add_port_to_graph(graph, instance_name, internal_port_name, directionality, external: bool, external_port_name=None, port_type=None):
    """Add one visual port node and connect it to its component node."""
    if external:
        node_name = f".{external_port_name}" # '.' enforces uniqueness
        shape="circle",
        size = 8
    else:
        node_name = f"{instance_name},{internal_port_name}" # ',' enforces uniqueness
        size = 8,
        shape="circle"

    graph.add_node(
            node_name, 
            shape=shape,
            opacity=1.0,
            border_color="black",
            border_size=1,
            size=size,
            color="white"
        )
    
    add_connection_to_graph(graph, node_name, instance_name.strip(), None, None, "bidirectional", directionality, port_type=port_type)

def instantiated_flat_netlist_to_graph(instantiated_flat_netlist, include_ports=False):
    """Convert an instantiated flat netlist into a graph.

    Unlike `netlist_to_graph`, this function reads port metadata from the
    instantiated component objects stored under each instance's `"model"` key.
    This is the graph form used by simulation drivers after `Circuit.instantiate`
    has expanded PCells and placeholders.
    """
    graph = nx.MultiDiGraph()
    # Add nodes for each instance
    for instance_name, instance in instantiated_flat_netlist["instances"].items():
        instantiated_flat_netlist['instances'][instance_name]['model']._create_port_lookup_table()
        graph.add_node(
            instance_name,
            component=instance["component"],
            settings=instance["settings"],
            shape="rectangle",
            size=25,
            border_size=1,
            border_color="black",
            color="white"
        )
        # graph.add_node(instance_name, label="test", click="Test: $label", **instance_data)
        # graph.add_node(instance_name, weight=netlist['instances'][instance_name]["weight"])


    # Add edges based on connections
    for src, dsts in instantiated_flat_netlist["connections"].items():
        for dst in dsts.split(";"):
            if dst =='':
                continue
            src_instance, src_port = src.split(",")
            dst_instance, dst_port = dst.split(",")
            
            src_port_directionality = instantiated_flat_netlist['instances'][src_instance]['model']._port_lookup_table[src_port].directionality
            dst_port_directionality = instantiated_flat_netlist['instances'][dst_instance]['model']._port_lookup_table[dst_port].directionality
            port_type = instantiated_flat_netlist['instances'][src_instance]['model']._port_lookup_table[src_port].type
            add_connection_to_graph(graph, src_instance.strip(), dst_instance.strip(), src_port.strip(), dst_port.strip(), src_port_directionality, dst_port_directionality, port_type=port_type)

    if include_ports:
        for external_port_name, internal_port_data in instantiated_flat_netlist["ports"].items():
            instance_name, internal_port_name = internal_port_data.split(",")
            node_name = f".{external_port_name}" # . Symbol Ensures Uniqueness of node_name
            model = instantiated_flat_netlist['instances'][instance_name]['model']
            directionality = model._port_lookup_table[internal_port_name].directionality
            port_type = model._port_lookup_table[internal_port_name].type
            add_port_to_graph(graph, instance_name, internal_port_name, directionality, external=True, external_port_name=external_port_name, port_type=port_type)

        unconnected_ports = set()
        connected_ports = {connection for connection in list(instantiated_flat_netlist['connections'].keys()) + list(instantiated_flat_netlist['connections'].values())}
        for instance_name, instance_data in instantiated_flat_netlist['instances'].items():
            for port in instantiated_flat_netlist['instances'][instance_name]['model'].ports:
                if not f"{instance_name},{port.name}" in connected_ports and not f"{instance_name},{port.name}" in instantiated_flat_netlist['ports'].values():
                    unconnected_ports.add(f"{instance_name},{port.name}")
                    # instantiated_flat_netlist['instances'][instance_name]['model']._port_lookup_table
                    add_port_to_graph(graph, instance_name, port.name, port.directionality, external=False, port_type=port.type)
                    # graph.add_node(
                    #     "FIX ME", 
                    #     shape="rectangle",
                    #     opacity=1.0,
                    #     border_color="black",
                    #     border_size=1,
                    #     size=15,
                    #     color="white"
                    # )
        pass
    
    #Matthew's Changes
    #Adds the ports as a graph attribute to be used within graph_to_netlist
    # graph.graph["ports"] = instantiated_flat_netlist.get("ports", {}).copy()
    ### FOR NOW, JUST BE OKAY WITH THE FACT THAT WE LOSE THIS INFO

    return graph

#Matthew's Changes
#Completed the graph to netlist to be used however needed to also added
import networkx as nx

def graph_to_netlist(graph: nx.MultiDiGraph, ports=None) -> dict:
    """Convert a NetworkX multidigraph into a SAX-compatible flat netlist.

    Assumptions:
    - Node attrs contain:
        - 'component'
        - optional 'settings'
    - Edge attrs contain:
        - 'src_port'
        - 'dst_port'

    Multiple edges between the same ports that merely represent
    bidirectionality are collapsed into a single connection.

    SAX connections are represented as:
        "inst1,portA": "inst2,portB"
    """

    if ports is None:
        ports = {}

    netlist = {
        "instances": {},
        "connections": {},
        "ports": ports.copy(),
    }

    # ------------------------------------------------------------------
    # Instances
    # ------------------------------------------------------------------

    for node, data in graph.nodes(data=True):
        netlist["instances"][node] = {
            "component": data["component"],
            "settings": data.get("settings", {}).copy(),
        }

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    # Use a set so we can ignore duplicated reverse-direction edges
    seen_connections = set()

    for src, dst, attrs in graph.edges(data=True):

        a = f"{src},{attrs['src_port']}"
        b = f"{dst},{attrs['dst_port']}"

        # Canonicalize connection ordering so:
        #   A -> B
        # and
        #   B -> A
        # are treated as the same physical connection
        canonical = tuple(sorted((a, b)))

        if canonical in seen_connections:
            continue

        seen_connections.add(canonical)

        # Store only one direction in SAX netlist
        netlist["connections"][a] = b

    return netlist
# def graph_to_netlist(graph: nx.MultiDiGraph, ports={}) -> dict:
#     """
#     Convert a NetworkX MultiDiGraph (with node attrs 'component' and 'settings',
#     edge attrs 'src_port' and 'dst_port', and graph.graph['ports']) back into a netlist:
#     """
#     netlist = {"instances": {}, "connections": {}}

#     for node, data in graph.nodes(data=True):
#         netlist["instances"][node] = {
#             "component": data["component"],
#             "settings":  data.get("settings", {}).copy()
#         }

#     conn_map = {}
#     for src, dst, attrs in graph.edges(data=True):
#         key  = f"{src},{attrs['src_port']}"
#         pair = f"{dst},{attrs['dst_port']}"
#         conn_map.setdefault(key, []).append(pair)

#     for key, dsts in conn_map.items():
#         netlist["connections"][key] = ";".join(dsts)

#     # ports = graph.graph.get("ports", {})
#     # netlist["ports"] = ports.copy()
#     netlist["ports"] = ports

#     return netlist

def sanitize_instance_names(netlist, old_separator="~", new_separator="_"):
    """Replace a separator substring in instance names and references.

    SAX flattening commonly creates nested instance names containing `"~"`.
    Some downstream tools expect valid Python identifiers, so this helper
    rewrites instance names and updates `connections`, `ports`, and optional
    `nets` entries consistently.
    """
    import copy

    netlist = copy.deepcopy(netlist)

    # mapping old instance names -> new names
    rename = {
        name: name.replace(old_separator, new_separator)
        for name in netlist.get("instances", {})
        if "~" in name
    }

    if not rename:
        return netlist

    def fix_ref(ref):
        """Fix 'instance,port' references."""
        if isinstance(ref, str) and "," in ref:
            inst, port = ref.split(",", 1)
            inst = rename.get(inst, inst)
            return f"{inst},{port}"
        return ref

    # ---- rename instances ----
    instances = netlist.get("instances", {})
    new_instances = {}
    for name, val in instances.items():
        new_instances[rename.get(name, name)] = val
    netlist["instances"] = new_instances

    # ---- fix connections ----
    if "connections" in netlist:
        new_connections = {}
        for k, v in netlist["connections"].items():
            new_connections[fix_ref(k)] = fix_ref(v)
        netlist["connections"] = new_connections

    # ---- fix ports ----
    if "ports" in netlist:
        netlist["ports"] = {
            name: fix_ref(ref)
            for name, ref in netlist["ports"].items()
        }

    # ---- fix nets (optional SAX format) ----
    if "nets" in netlist:
        for net in netlist["nets"]:
            net["p1"] = fix_ref(net["p1"])
            net["p2"] = fix_ref(net["p2"])

    return netlist

# def discrete_time_impulse_response(propagation_constants, sampling_freq, length=1e-6, N=20000):
#     freqs = jnp.fft.fftfreq(N, d=1/sampling_freq)
#     omega = 2*jnp.pi*freqs
#     phi = jnp.zeros_like(omega, dtype=jnp.complex64)

#     for k, beta_k in propagation_constants.items():
#         phi += (beta_k * length * (omega ** k)) / factorial(k)

#     H = jnp.exp(-1j * phi)

#     return jnp.fft.ifftshift(jnp.fft.ifft(H))

def generate_valid_separator(instance_names, old_separator="~", first_try="_SEP_"):
    """Generate a separator that will not collide when replacing names.

    The returned separator can be used to replace `old_separator` without
    causing two distinct instance names to collapse to the same string.
    """
    instance_names = list(instance_names)
    new_separator = first_try
    new_instance_names = [name.replace(old_separator, new_separator) for name in instance_names]
    while not len(set(instance_names)) == len(set(new_instance_names)):
        new_separator = "_" + new_separator + "_"
        new_instance_names = [name.replace(old_separator, new_separator) for name in instance_names]
    
    return new_separator

def generate_unique_string(instance_names, first_try="xXx"):
    """Return a string that is not contained in any instance name."""
    instance_names = list(instance_names)
    new_str = first_try

    def instances_contain_string(str):
        for instance_name in instance_names:
            if str in instance_name:
                return True
        return False

    while instances_contain_string(new_str):
        new_str = "_" + new_str + "_"
    
    return new_str
    

def remove_instances_from_netlist(netlist, instances_to_remove):
    """Return a copy of `netlist` without selected instances.

    Connections and top-level ports touching removed instances are removed as
    well. This is useful when extracting or replacing subgraphs during PCell and
    S-parameter consolidation.
    """
    instances_to_remove = set(instances_to_remove)

    new_netlist = {
        "instances": {},
        "connections": {},
        "ports": {},
    }

    # Keep surviving instances
    new_netlist["instances"] = {
        name: inst
        for name, inst in netlist["instances"].items()
        if name not in instances_to_remove
    }

    def touches_removed_instance(endpoint):
        instance_name = endpoint.split(",")[0]
        return instance_name in instances_to_remove

    # Keep only valid connections
    new_netlist["connections"] = {
        src: dst
        for src, dst in netlist["connections"].items()
        if not (
            touches_removed_instance(src)
            or touches_removed_instance(dst)
        )
    }

    # Keep only valid external ports
    new_netlist["ports"] = {
        port: endpoint
        for port, endpoint in netlist["ports"].items()
        if not touches_removed_instance(endpoint)
    }

    return new_netlist
