# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simphony.circuit import Circuit
from copy import deepcopy
from dataclasses import field
from functools import partial

import networkx as nx

# from simphony.utils import graph_to_netlist
import sax
from flax import struct
from jax.typing import ArrayLike

from simphony.circuit.circuit import Circuit
from simphony.circuit.netlist import (
    generate_valid_separator,
    graph_to_netlist,
    sanitize_instance_names,
)
from simphony.component.component import SParameterComponent

from .simulation import (
    Simulation,
    SimulationMode,
    SimulationParameters,
    SimulationResult,
)
from .steady_state import SteadyStateSimulation, SteadyStateSimulationParameters


@struct.dataclass
class SParameterSimulationParameters(SimulationParameters):
    """Global settings for S-parameter simulations.

    S-parameter simulations operate in the frequency domain and expose a
    SAX `SDict` over selected circuit ports.
    """

    simulation_mode: SimulationMode = field(
        default_factory=lambda: SimulationMode.S_PARAMETER
    )
    # optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jnp.array([1.54e-6, 1.55e-6, 1.56e-6]))
    directed: bool = False


class SParameterSimulationResult(SimulationResult):
    """Result returned by `SParameterSimulation.run`.

    Attributes are attached by `run`:

    - `sax_circuit`: callable SAX circuit assembled for the requested ports.
    - `sax_circuit_info`: metadata returned by `sax.circuit`.
    - `s_parameters`: evaluated `SDict` at the requested wavelength grid.
    """

    def __init__(self):
        pass


class SParameterSimulation(Simulation):
    """Compute frequency-domain S-parameters for selected circuit ports.

    The simulator instantiates the circuit, separates the optical S-parameter
    subgraph from any steady-state bias circuitry, computes required bias
    values, and builds a SAX circuit for the requested ports.

    Parameters
    ----------
    circuit:
        Circuit to simulate.
    settings:
        Per-instance settings used during instantiation.
    simulation_parameters:
        Shared `SParameterSimulationParameters`. Defaults are used when omitted.
    ports:
        Optional mapping of exposed port names to top-level port designators. If
        omitted, the circuit top-level ports are used.
    """

    def __init__(
        self,
        circuit: Circuit,
        settings,
        simulation_parameters=None,
        ports=None,
        # settings: dict = None
    ):
        """s_parameter_simulation Calculates the S-parameters for a given set
        of ports in an optical circuit.

        By default, the exposed ports and settings are taken from the
        provided netlist, but may be overwritten with keyword arguments.
        """
        if simulation_parameters is None:
            simulation_parameters = SParameterSimulationParameters()

        self.circuit = circuit
        # self.flat_circuit = circuit.flatten()
        self.settings = settings
        self.simulation_parameters = simulation_parameters

        if ports is None:
            ports = self.circuit.netlist["top_level"]["ports"]

        self.ports = ports

    def run(
        self,
        wl: ArrayLike = 1.55e-6,
        # use_default_settings: bool = True
    ) -> SParameterSimulationResult:
        """Evaluate the circuit S-parameters at wavelength `wl`.

        Parameters
        ----------
        wl:
            Wavelength or wavelength array, in meters.

        Returns
        -------
        SParameterSimulationResult
            Result containing the generated SAX circuit and evaluated `SDict`.
        """
        self.instantiated_circuit = self.circuit.instantiate(
            self.settings, self.simulation_parameters
        )

        # self._identify_component_types()
        (
            steady_state_graph,
            reachable_bias_nodes,
            s_parameter_graph,
        ) = self._build_s_parameter_circuit(self.ports)
        # self._validate_s_parameter_graph()

        # TODO: FIX THE FACT THAT PORTS DONT GET CONVERETED RIGHT IN graph_to_netlist
        steady_state_netlist = graph_to_netlist(steady_state_graph)
        steady_state_models = {
            data["component"]: type(data["model"])
            for instance_name, data in self.instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ].items()
            if instance_name in steady_state_netlist["instances"]
        }

        # TODO: Verify that we are able to obtain the settings from nested subcircuits / pcells
        steady_state_circuit_settings = {
            instance_name: data["settings"]
            for instance_name, data in self.instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ].items()
            if instance_name in steady_state_netlist["instances"]
        }
        # TODO: FIX CIRCUIT SO THAT IT WORKS WITH SINGLE ELEMENT NETLISTS
        steady_state_circuit = Circuit(steady_state_netlist, steady_state_models)

        steady_state_simulation_parameters = SteadyStateSimulationParameters()
        steady_state_simulation = SteadyStateSimulation(
            steady_state_circuit,
            steady_state_circuit_settings,
            steady_state_simulation_parameters,
        )
        # self._initialize_steady_state_simulation(s_parameter_graph, steady_state_graph)
        # self.reset_settings(use_default_settings=True)

        s_parameter_simulation_result = SParameterSimulationResult()
        use_default_settings = True
        # self.reset_settings(use_default_settings=use_default_settings)
        # self.add_settings(settings)
        # s_parameter_result = SParameterSimulationResult()

        # self._instantiate_components(self.settings)
        steady_state_simulation_result = steady_state_simulation.run()
        ports = {
            k: self.instantiated_circuit.port_lookup_table[k] for k in self.ports.keys()
        }
        sax_circuit, sax_circuit_info = self._generate_sax_circuit(
            s_parameter_graph,
            steady_state_simulation_result,
            reachable_bias_nodes,
            ports,
        )
        s_parameter_simulation_result.sax_circuit = sax_circuit
        s_parameter_simulation_result.sax_circuit_info = sax_circuit_info
        s_parameter_simulation_result.s_parameters = sax_circuit(wl=wl)
        return s_parameter_simulation_result

    def _identify_component_types(self):
        """Identify the types of components in the circuit.

        This method categorizes components into electrical, optical, and
        logic components
        """
        self.all_components = set()
        self.electrical_components = set()
        self.optical_components = set()
        self.logic_components = set()

        graph = self.instantiated_circuit.graph

        for node, attr in graph.nodes(data=True):
            component = self.instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ][node]["model"]

        # models = self.instantiated_circuit.models
        # for node, attr in graph.nodes(data=True):
        #     model = attr["component"]
        #     component = models[model]

        #     self.all_components.add(node)

        #     component_types = attr['type'].lower().split('/')
        #     if "electrical" in component_types:
        #         self.electrical_components.add(node)
        #     if "optical" in component_types:
        #         self.optical_components.add(node)
        #     if "logic" in component_types:
        #         self.logic_components.add(node)

        # if component.electrical_ports:
        #     self.electrical_components.add(node)
        # if component.logic_ports:
        #     self.logic_components.add(node)
        # if component.optical_ports:
        #     self.optical_components.add(node)

    def _build_s_parameter_circuit(self, ports: dict):
        """Split the instantiated graph into optical and bias subgraphs.

        The S-parameter simulator needs two pieces of information:

        - the connected optical S-parameter subgraph reachable from the exposed
          ports, which becomes a SAX circuit; and
        - any steady-state bias/control components that drive S-parameter bias
          ports.

        Returns
        -------
        tuple
            `(steady_state_graph, reachable_bias_nodes, s_parameter_graph)`.
            `reachable_bias_nodes` maps bias-source designators to the
            S-parameter instance and port they control.
        """
        # Step 1: Create a new graph with only s-parameter components
        s_parameter_nodes = []
        for node, data in self.instantiated_circuit.graph.nodes(data=True):
            model = self.instantiated_circuit.instantiated_flat_netlist["instances"][
                node
            ]["model"]
            if isinstance(model, SParameterComponent):
                s_parameter_nodes.append(node)

        s_parameter_only_graph = self.instantiated_circuit.graph.subgraph(
            s_parameter_nodes
        ).copy()
        # all_optical_graph = deepcopy(self.instantiated_circuit.graph)

        # optical_bias_ports = {}
        # for src_node, dst_node, key, data in self.instantiated_circuit.graph.edges(keys=True, data=True):
        #     ### TODO: Write tests for whether this works in all edge cases
        #     dst_port = data['dst_port']
        #     dst_bias_ports = self.instantiated_circuit.instantiated_flat_netlist['instances'][dst_node]['model']._s_parameter_get_bias_ports()
        #     src_port = data['src_port']
        #     src_bias_ports = self.instantiated_circuit.instantiated_flat_netlist['instances'][src_node]['model']._s_parameter_get_bias_ports()
        #     if not data['port_type'] == 'optical' or dst_port in dst_bias_ports:
        #         all_optical_graph.remove_edge(src_node, dst_node, key)

        # if dst_port in dst_bias_ports:
        #     all_optical_graph.remove_edge(src_node, dst_node, key)
        #     optical_bias_ports[dst_node] = dst_port

        # TODO: REMOVE bias port connections and verify

        reachable = set()
        for ext_port, (instance_port) in ports.items():
            port_designator = self.instantiated_circuit.port_lookup_table[ext_port]
            instance_name = port_designator.split(",")[0]
            # print(nx.descendants(all_optical_graph, port_designator))
            # print(nx.ancestors(all_optical_graph, port_designator))
            reachable.add(instance_name)
            reachable |= nx.descendants(s_parameter_only_graph, instance_name)
            reachable |= nx.ancestors(s_parameter_only_graph, instance_name)

        s_parameter_graph = s_parameter_only_graph.subgraph(reachable).copy()
        reachable_bias_nodes = {}

        # TODO: add this functionality to instantiated flat netlist or instantiated circuit
        flipped_connections = {
            v: k
            for k, v in self.instantiated_circuit.instantiated_flat_netlist[
                "connections"
            ].items()
        }
        for node in s_parameter_graph.nodes():
            ### TODO: Write tests for whether this works in all edge cases
            bias_ports = self.instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ][node]["model"]._s_parameter_get_bias_ports()

            for bias_port in bias_ports:
                bias_ports

                if f"{node},{bias_port}" in flipped_connections:
                    bias_node, port_name = flipped_connections[
                        f"{node},{bias_port}"
                    ].split(",")
                    reachable_bias_nodes[(bias_node, port_name)] = (node, bias_port)

        # Check is an ancestor of the bias node is in the s_parameter graph
        # Check whether all ancestors are SteadyStateComponents
        # TODO: eliminate all non-bias connections first
        steady_state_nodes = set()
        for (biasing_node, biasing_port), (
            biased_node,
            biased_port,
        ) in reachable_bias_nodes.items():
            steady_state_nodes.add(biasing_node)
            for ancestor in nx.ancestors(self.instantiated_circuit.graph, biasing_node):
                steady_state_nodes.add(ancestor)
                # TODO: Check if ancestor is in s_parameter graph and throw an error

        steady_state_graph = self.instantiated_circuit.graph.subgraph(
            list(steady_state_nodes)
        ).copy()
        # nx.connected_components(steady_state_only_graph)
        return steady_state_graph, reachable_bias_nodes, s_parameter_graph

    def _validate_s_parameter_graph(self):
        # Signal source nodes are sources of non-optical signals
        source_nodes = set()
        s_parameter_graph_nodes = set(self.s_parameter_circuit.graph.nodes)
        potential_source_nodes = (
            s_parameter_graph_nodes
            & self.optical_components
            & (self.electrical_components | self.logic_components)
        )
        for node in potential_source_nodes:
            out_edges = self.circuit.graph.out_edges(node, data=True)
            for src, dst, attr in out_edges:
                src_port = attr["src_port"]
                if src_port not in self.circuit.graph.nodes[src]["optical ports"]:
                    source_nodes.add(node)

        # We do not allow any of the s_parameter_nodes to function as
        # signal sources that feed back into the s_parameter_nodes
        # Such simulations should be performed in the time-domain
        non_s_parameter_graph = deepcopy(self.circuit.graph)
        non_s_parameter_graph.remove_edges_from(self.s_parameter_circuit.graph.edges())
        for source_node in source_nodes:
            descendants = nx.descendants(non_s_parameter_graph, source_node)
            if len(descendants & s_parameter_graph_nodes) > 0:
                raise ValueError(
                    "Invalid S-parameter SubCircuit: Time-domain Simulation Required"
                )

    # def _initialize_steady_state_simulation(self, s_parameter_graph, steady_state_graph):
    #     steady_state_circuit = deepcopy(self.circuit)
    #     steady_state_circuit.remove_components(self.s_parameter_circuit.graph.nodes-self.hybrid_components)
    #     self.steady_state_simulation = SteadyStateSimulation(steady_state_circuit)
    #     # steady_state_graph.remove_nodes_from(self.s_parameter_graph.nodes)
    #     # self.steady_state_simulation = SteadyStateSimulation(self.steady_state_graph)

    ### I am going to put this in the base class
    # def _instantiate_components(self):
    #     self.components = {}
    #     for component_name in self.circuit.graph.nodes:
    #         model_name = self.circuit.netlist['instances'][component_name]['component']
    #         model = self.circuit.models[model_name]
    #         settings = self.settings[component_name]
    #         self.components[component_name] = model(**settings)

    # def _calculate_steady_states(self):
    #     for component in self.steady_state_order:
    #         pass

    def _generate_sax_circuit(
        self,
        s_parameter_graph,
        steady_state_simulation_result,
        reachable_bias_nodes,
        ports,
    ):
        """Build a callable SAX circuit from the reachable optical subgraph.

        Bias-dependent S-parameter components are partially applied with the
        steady-state signals computed earlier in `run`. The resulting callable
        accepts the usual SAX wavelength argument and returns an `SDict` over the
        requested exposed ports.

        Parameters
        ----------
        s_parameter_graph:
            Reachable graph containing only S-parameter components.
        steady_state_simulation_result:
            Result containing bias/control outputs for any reachable bias ports.
        reachable_bias_nodes:
            Mapping produced by `_build_s_parameter_circuit`.
        ports:
            Exposed SAX port mapping for the generated circuit.
        """
        # I will assume that the only connections between the s-parameter portion of the circuit
        # and the steady-state portion of the circuit are electrical or optical (this might change)
        # in the future if more connection types become supported

        # These components will need to have their s_parameter methods completed with the steady state inputs

        # TODO: Test this for multiple connections
        incomplete_components = {
            node: {} for (node, port) in reachable_bias_nodes.values()
        }
        for source, (node, port) in reachable_bias_nodes.items():
            incomplete_components[node][port] = source
        component_inputs = {component: {} for component in s_parameter_graph}

        for (
            incomplete_component,
            biasing_node_designator,
        ) in incomplete_components.items():
            bias_nodes = incomplete_components[incomplete_component]
            for dst_port, (src_node, src_port) in biasing_node_designator.items():
                component_inputs[incomplete_component][
                    dst_port
                ] = steady_state_simulation_result.component_outputs[src_node][src_port]

            # component_inputs[incomplete_component] = steady_state_simulation_result.component_inputs[incomplete_component]

        # sax_models
        separator = generate_valid_separator(component_inputs.keys())
        sax_models = {}
        for component, inputs in component_inputs.items():
            # model_name = self.circuit.netlist['instances'][component]['component']
            s_parameter_func = self.instantiated_circuit.instantiated_flat_netlist[
                "instances"
            ][component]["model"].s_parameters
            # TODO: Make sure i am replacing with a unique value ("_" does not guarantee a unique instance name)
            sax_models[component.replace("~", separator)] = partial(
                s_parameter_func, inputs
            )

        # Each instance should correspond to a unique model at this point
        # Since they might have been changed but the steady state inputs
        # Therefore, we need the netlist to reflect this change

        # instances = {key: key for key in self.s_parameter_circuit.netlist['instances']}
        # self.s_parameter_circuit.netlist['instances'] = instances
        s_parameter_netlist = graph_to_netlist(s_parameter_graph)
        for instance_name, data in s_parameter_netlist["instances"].items():
            s_parameter_netlist["instances"][instance_name] = instance_name.replace(
                "~", separator
            )
            # data['component'] = instance_name.replace("~", "_")

        s_parameter_netlist["ports"] = ports
        # s_parameter_netlist['ports'] = {"in":"combiner~sax_model,port_1", "out": "splitter~sax_model,port_1"}
        s_parameter_netlist = sanitize_instance_names(
            s_parameter_netlist, new_separator=separator
        )

        # s_parameter_func = self.instantiated_circuit.instantiated_flat_netlist['instances']["wg1~sax_model"]['model'].s_parameters(np.array([1.55]), {})
        return sax.circuit(s_parameter_netlist, sax_models)
        """### TODO: MATTHEW! Keep in mind that I defined the sax models to use
        SI units ### and to assume that wl is given in terms of meters, not
        microns ### To see what I mean, stop here in debug mode and run
        sax_models['splitter'](1.55e-6) ### Notice that I am using 1.55e-6
        instead of 1.55 but that is what it expects pass

        ### TODO: complete the function graph_to_netlist in simphony.utils
        ### turn self.s_parameter_graph into a netlist
        ### use the sax_models dictionary above and the netlist you just generated to create
        ### a sax.circuit and return the resulting scattering dictionary
        ### Alternatively, it might be more efficient to make an "self.s_parameter_circuit"
        ### and use the remove_compoenents method from that circuit object
        ### that might get you the netlist you need for free

        sax_netlist = graph_to_netlist(self.s_parameter_circuit.graph) ## You might change this to use the alternative approach
        circuit = sax.circuit(sax_netlist, sax_models)
        ### TODO: It is probably possible for the user to keep the s-parameter graph elements parameterized
        ### and simply return a sax circuit, maybe I will do that later, don't do that yet,
        ### For now just return the s-parameter dict
        return circuit(wl)
        """
