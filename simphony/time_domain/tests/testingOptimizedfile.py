import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from jax import config

config.update("jax_enable_x64", True)

from simphony.libraries import ideal, siepic
from simphony.time_domain.pole_residue_model import IIRModelBaseband
from simphony.time_domain.utils import gaussian_pulse, pole_residue_to_time_system


def remove_active_connections(connections, active_components):
    """
    Remove all connections that involve any active component.
    """
    filtered_connections = {}
    for k, v in connections.items():
        compA, portA = k.split(",")
        compB, portB = v.split(",")

        # Skip if either endpoint is an active component
        if compA in active_components or compB in active_components:
            continue

        # Otherwise keep it
        filtered_connections[k] = v
    return filtered_connections


def remove_active_ports(ports, active_components):
    """
    Remove ports that connect to active components.
    """
    filtered_ports = {}
    for port_label, comp_port_str in ports.items():
        comp, cport = comp_port_str.split(",")
        if comp not in active_components:
            filtered_ports[port_label] = comp_port_str
    return filtered_ports


def build_graph(connections, directed=False):
    """
    Build an adjacency list from the netlist connections.
    Each (component, port) pair is treated as a graph node.

    If directed=False, each connection is stored as two directed edges (A->B, B->A).
    If directed=True, only the forward direction is stored.
    """
    graph = {}

    def add_edge(node1, node2):
        if node1 not in graph:
            graph[node1] = []
        graph[node1].append(node2)

    for k, v in connections.items():
        compA, _ = k.split(",")
        compB, _ = v.split(",")

        nodeA = compA
        nodeB = compB

        # Always add A->B
        add_edge(nodeA, nodeB)
        # If it's undirected, also add B->A
        if not directed:
            add_edge(nodeB, nodeA)

    return graph


def tarjan_scc(graph):
    """
    Tarjan's SCC algorithm.
    Returns a list of strongly connected components (SCCs),
    where each SCC is a list of nodes [(comp, port), ...].
    """
    index_counter = [0]  # mutable counter
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

        # Consider successors of node
        for w in graph.get(node, []):
            if w not in indices:
                # Successor w has not yet been visited; recurse on it
                strongconnect(w)
                lowlinks[node] = min(lowlinks[node], lowlinks[w])
            elif w in on_stack:
                # Successor w is in the stack and hence in the current SCC
                lowlinks[node] = min(lowlinks[node], indices[w])

        # If node is a root node, pop the stack and generate an SCC
        if lowlinks[node] == indices[node]:
            # Start a new SCC
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    # Run strongconnect for each node that is unvisited
    for node in graph.keys():
        if node not in indices:
            strongconnect(node)

    return sccs


def find_sccs_for_passive_subnets(
    instances, connections, ports, active_components, directed=False
):
    """
    Main function that orchestrates:
      1) Removal of active-component connections
      2) Graph construction
      3) Tarjan's SCC search
      4) Grouping the results by component (optional)

    Returns:
      A list of sets, each set containing the (passive) components
      that form a strongly connected sub-net (SCC).
    """

    # 1. Remove connections that involve active components
    filtered_conns = remove_active_connections(connections, active_components)

    # 2. Remove top-level ports referencing active components (optional)
    filtered_ports = remove_active_ports(ports, active_components)

    # 3. Build a graph from the remaining passive connections
    graph = build_graph(filtered_conns, directed=directed)

    # 4. Run Tarjan’s to find SCCs
    sccs = tarjan_scc(graph)

    # 5. Group by component (optional). If you want per-port detail, skip this step.
    # scc_groups = group_components_by_scc(sccs)

    return sccs


# -------------------------
# Example usage:
if __name__ == "__main__":
    # Example netlist data
    instances = {
        "cr1": "y_branch",
        "cr2": "y_branch",
        "wg1": "waveguide",
        "wg2": "waveguide",
        "cr3": "y_branch",
        "cr4": "y_branch",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "pm": "phase_modulator",
        "cr5": "y_branch",
        "cr6": "y_branch",
        "wg5": "waveguide",
        "tm": "tunable_modulator",
    }
    connections = {
        "tm,o0": "cr1,port_2",
        "tm,o1": "wg1,o1",
        "cr1,port_3": "wg2,o0",
        "wg1,o1": "cr2,port_2",
        "wg2,o1": "cr2,port_3",
        "cr2, port_1": "cr5,port_1",
        "cr3, port_1": "cr6,port_1",
        "cr3,port_2": "wg3,o0",
        "cr3,port_3": "wg4,o0",
        "cr4,port_2": "wg3,o1",
        "cr4,port_3": "wg4,o1",
        "cr5,port_2": "wg5,o0",
        "cr5,port_3": "pm,o0",
        "cr6,port_2": "wg5,o1",
        "cr6,port_3": "pm,o1",
    }
    ports = {
        "o0": "cr1,port_1",
        "o1": "cr4,port_1",
    }
    active_components = {"pm", "tm"}

    # Find the passive SCC groups
    passive_sccs = find_sccs_for_passive_subnets(
        instances, connections, ports, active_components, directed=False
    )

    print("Passive strongly connected sub-net groups (by component):")
    for group in passive_sccs:
        print("  ", group)
