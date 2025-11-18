import inspect

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

# from simphony.utils import dict_to_matrix

COMPONENT_COLOR_DEFAULT = "black"
COMPONENT_COLOR_SPARAM = "blue"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"
COMPONENT_COLOR_LOGIC = "gray"

"""
Todo: Give S-parameter elements proper abstraction
"""
# class SParameterComponent(Component):
#     pass


class Circuit:
    def __init__(
        self,
        netlist: dict,
        models: dict,
        default_settings: dict = None
    ) -> None:
        # if settings is not None:
        #     add_settings_to_netlist(netlist, settings)
        # else:
        #     add_settings_to_netlist(netlist, None)
        self.netlist = deepcopy(netlist)
        if not 'ports' in self.netlist:
            self.netlist['ports'] = {}
        add_settings_to_netlist(self.netlist, default_settings)
        self.default_settings = get_settings_from_netlist(self.netlist) 
        self.models = models
        # self.settings = settings
        
        self.graph = netlist_to_graph(self.netlist)
        
        self._convert_sax_models()
        # self._add_models_to_graph(self.models)
        self._mark_component_types()
        self._add_ports_to_graph()
        self._validate_connections()
        self._color_nodes()

    def display(self, inline=True):
        fig = gv.d3(self.graph)
        fig.display(inline=inline)
    

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

    def _convert_sax_models(self):
        for model in self.models:
            component = self.models[model]
            if not inspect.isclass(component):
                s_parameter = optical_s_parameter(component)
                self.models[model] = s_parameter

    def _mark_component_types(self):
        """ 
        """
        for instance_name, attr in self.graph.nodes.items():
            component_name = attr["component"]
            component = self.models[component_name]

            tags = set()
            for port in component.ports:
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

            self.graph.nodes[instance_name]["type"] = "/".join(sorted(tags))

    def _add_ports_to_graph(self):
        for instance_name, attr in self.graph.nodes.items():
            self.graph.nodes[instance_name]["electrical ports"] = []
            self.graph.nodes[instance_name]["logic ports"] = []
            self.graph.nodes[instance_name]["optical ports"] = []

            model = attr["component"]
            component = self.models[model]

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
            
            self.graph.nodes[instance_name]["electrical ports"] = electrical_ports
            self.graph.nodes[instance_name]["optical ports"] = optical_ports
            self.graph.nodes[instance_name]["logic ports"] = logic_ports

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

    def get_port_type(self, instance, port):
        optical_ports = self.graph.nodes[instance]["optical ports"]
        electrical_ports = self.graph.nodes[instance]["electrical ports"]
        logic_ports = self.graph.nodes[instance]["logic ports"]

        if port in optical_ports:
            return "optical"
        elif port in electrical_ports:
            return "electrical"
        elif port in logic_ports:
            return "logic"

    def _validate_connections(self):
        # Verify optical-to-optical, electrical-to-electrical, logic-to-logic
        for edge in self.graph.edges:
            src, dst, _ = edge
            
            src_port = self.graph.edges[edge]["src_port"]
            src_port_type = self.get_port_type(src, src_port)

            dst_port = self.graph.edges[edge]["dst_port"]
            dst_port_type = self.get_port_type(dst, dst_port)

            if not src_port_type == dst_port_type:
                raise ValueError("Port types must match")
        
        # TODO: Verify out to in or out to bidirectional connections

    def _color_nodes(self):
        color = COMPONENT_COLOR_DEFAULT

        # Assumes tags in alphabetical order
        for instance in self.graph.nodes:
            if self.graph.nodes[instance]["type"] == "s-parameter":
                color = COMPONENT_COLOR_SPARAM
            elif self.graph.nodes[instance]["type"] == "electrical/optical":
                color = COMPONENT_COLOR_OPTOELECTRICAL
            elif self.graph.nodes[instance]["type"] == "optical":
                color = COMPONENT_COLOR_OPTICAL
            elif self.graph.nodes[instance]["type"] == "electrical":
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]["type"] == "logic":
                color = COMPONENT_COLOR_LOGIC
            elif self.graph.nodes[instance]["type"] == "electrical/logic":
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]["type"] == "logic/optical":
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]["type"] == "electrical/logic/optical":
                color = COMPONENT_COLOR_OPTOELECTRICAL

            self.graph.nodes[instance]["color"] = color
