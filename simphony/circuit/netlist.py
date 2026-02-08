from sax import AnyNetlist, InstanceName, Ports, Connections, Models
import sax
from typing import TypeAlias, TypedDict
from simphony.component.component import Component
from typing_extensions import NotRequired
import networkx as nx
import jax.numpy as jnp
import yaml
from jax.scipy.special import factorial
from typing import Union
from copy import deepcopy
from simphony.component.pcell import PCell
from simphony.circuit._netlist import _instantiate_netlist, _add_settings_to_netlist, InstantiatedFlatNetlist, ElaboratedInstances
from simphony.libraries.ideal.s_parameters import optical_s_parameter
from simphony.simulation.simulation import SimulationParameters
from sax import DEFAULT_MODES
import inspect

def instantiate_netlist(
    netlist: dict,
    models: dict,
    settings: dict,
    simulation_parameters: SimulationParameters,
    # directed: bool = False,
    # default_modes: tuple = DEFAULT_MODES,
)->InstantiatedFlatNetlist:
    """
    parameters
    directed: whether sax models should be interpreted as directed or not
    default_modes: default_modes for sax models
    Will automatically walk down the tree to flatten PCell structures. 
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
    """
    Netlists may not have ports, or connections specified. 
    This function adds empty dictionaries to flat netlists
    without these specified.

    """
    if 'instances' not in netlist:
        netlist['instances'] = {}
    if 'connections' not in netlist:
        netlist['connections'] = {}
    if 'ports' not in netlist:
        netlist['ports'] = {}


def add_settings_to_netlist(netlist, settings=None):
    return _add_settings_to_netlist(netlist, settings=settings)

def get_settings_from_netlist(netlist):
    settings = {}
    for instance, attr in netlist['instances'].items():
        settings[instance] = attr['settings']
    
    return settings


def netlist_to_graph(netlist: Union[dict, str], models):
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

    # graph = nx.Graph()
    # graph = nx.MultiGraph()
    graph = nx.MultiDiGraph()
    # Add nodes for each instance
    for instance_name, instance in netlist["instances"].items():
        graph.add_node(
            instance_name,
            component=instance["component"],
            settings=instance["settings"],
            # image="image.png",
            # opacity=0.5,
            # size=5,
        )
        models[netlist['instances'][instance_name]['component']]._create_port_lookup_table()
        # graph.add_node(instance_name, label="test", click="Test: $label", **instance_data)
        # graph.add_node(instance_name, weight=netlist['instances'][instance_name]["weight"])
    
    # for port_name, port_data in netlist["ports"].items():
    #     instance_name, instance_port_name = port_data.split(",")
    #     graph.add_node(
    #         f"Kablooey-{port_name}",
    #         # component=instance["component"],
    #         # settings=instance["settings"],
    #         shape="rectangle",
    #         opacity=0.1,
    #         border_color="black",
    #         border_size=5,
    #         # image="image.png",
    #         # opacity=0.5,
    #         # size=5,
    #     )

    # Add edges based on connections
    for src, dsts in netlist["connections"].items():
        for dst in dsts.split(";"):
            if dst =='':
                continue
            src_instance, src_port = src.split(",")
            dst_instance, dst_port = dst.split(",")
            src_port_directionality = models[netlist['instances'][instance_name]['component']]._port_lookup_table[src_port].directionality
            dst_port_directionality = models[netlist['instances'][instance_name]['component']]._port_lookup_table[dst_port].directionality
            add_edge_to_graph(graph, src_instance.strip(), dst_instance.strip(), src_port.strip(), dst_port.strip(), src_port_directionality, dst_port_directionality)
            # graph.add_edge(src_instance.strip(), dst_instance.strip(), src_port=src_port.strip(), dst_port=dst_port.strip())
            
    
    #Matthew's Changes
    #Adds the ports as a graph attribute to be used within graph_to_netlist
    graph.graph["ports"] = netlist.get("ports", {}).copy()

    return graph

def add_edge_to_graph(graph, src_node, dst_node, src_port, dst_port, src_directionality, dst_directionality):
    if (src_directionality == "bidirectional" and dst_directionality == "bidirectional"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}↔{dst_node},{dst_port}")
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}↔{dst_node},{dst_port}")
    elif (src_directionality == "bidirectional" and dst_directionality == "input") or (src_directionality == "output" and dst_directionality == "bidirectional") or (src_directionality == "output" and dst_directionality == "input"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}→{dst_node},{dst_port}")
    elif (src_directionality == "bidirectional" and dst_directionality == "output") or (src_directionality == "input" and dst_directionality == "bidirectional") or (src_directionality == "input" and dst_directionality == "output"):
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}←{dst_node},{dst_port}")
    elif (src_directionality == "unknown" and dst_directionality== "input") or (src_directionality == "output" and dst_directionality== "unknown"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}→{dst_node},{dst_port}")
    elif (src_directionality == "unknown" and dst_directionality== "output") or (src_directionality == "input" and dst_directionality== "unknown"):
        graph.add_edge(dst_node, src_node, src_port=dst_port, dst_port=src_port, hover=f"{src_node},{src_port}←{dst_node},{dst_port}")
    elif (src_directionality == "unknown" or dst_directionality== "unknown"):
        graph.add_edge(src_node, dst_node, src_port=src_port, dst_port=dst_port, hover=f"{src_node},{src_port}?⎯?{dst_node},{dst_port}", color="red")
    else:
        raise ValueError(f"Cannot connect {src_directionality} to {dst_directionality}")

def instantiated_flat_netlist_to_graph(instantiated_flat_netlist):
    graph = nx.MultiDiGraph()
    # Add nodes for each instance
    for instance_name, instance in instantiated_flat_netlist["instances"].items():
        instantiated_flat_netlist['instances'][instance_name]['model']._create_port_lookup_table()
        graph.add_node(
            instance_name,
            component=instance["component"],
            settings=instance["settings"],
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
            
            add_edge_to_graph(graph, src_instance.strip(), dst_instance.strip(), src_port.strip(), dst_port.strip(), src_port_directionality, dst_port_directionality)
            
    
    #Matthew's Changes
    #Adds the ports as a graph attribute to be used within graph_to_netlist
    graph.graph["ports"] = instantiated_flat_netlist.get("ports", {}).copy()

    return graph

#Matthew's Changes
#Completed the graph to netlist to be used however needed to also added
def graph_to_netlist(graph: nx.MultiDiGraph) -> dict:
    """
    Convert a NetworkX MultiDiGraph (with node attrs 'component' and 'settings',
    edge attrs 'src_port' and 'dst_port', and graph.graph['ports']) back into a netlist:
    """
    netlist = {"instances": {}, "connections": {}}

    for node, data in graph.nodes(data=True):
        netlist["instances"][node] = {
            "component": data["component"],
            "settings":  data.get("settings", {}).copy()
        }

    conn_map = {}
    for src, dst, attrs in graph.edges(data=True):
        key  = f"{src},{attrs['src_port']}"
        pair = f"{dst},{attrs['dst_port']}"
        conn_map.setdefault(key, []).append(pair)

    for key, dsts in conn_map.items():
        netlist["connections"][key] = ";".join(dsts)

    ports = graph.graph.get("ports", {})
    netlist["ports"] = ports.copy()

    return netlist

# def discrete_time_impulse_response(propagation_constants, sampling_freq, length=1e-6, N=20000):
#     freqs = jnp.fft.fftfreq(N, d=1/sampling_freq)
#     omega = 2*jnp.pi*freqs
#     phi = jnp.zeros_like(omega, dtype=jnp.complex64)

#     for k, beta_k in propagation_constants.items():
#         phi += (beta_k * length * (omega ** k)) / factorial(k)

#     H = jnp.exp(-1j * phi)

#     return jnp.fft.ifftshift(jnp.fft.ifft(H))