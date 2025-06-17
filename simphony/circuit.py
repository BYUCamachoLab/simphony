import inspect

# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
import gravis as gv
import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

from simphony.utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph
from copy import deepcopy

class Component:
    electrical_ports = []
    logic_ports = []
    optical_ports = []

    def __init__(self, **kwargs):
        self.settings = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class SpectralSystem(Component):
    """ 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def steady_state(
        self, 
        inputs: dict
        ) -> dict:
        """
        Used when calculating steady state voltages for SParameterSimulation
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )

class OpticalSParameter(SpectralSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def s_parameters(
        self, 
        wl: ArrayLike, 
        # **kwargs
    ):
        """
        Returns an S-parameter matrix for the optical ports in the system
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )




    # def dc_voltage(self, **kwargs):
    #     """
    #     Returns the dc voltage for each electrical port in the system
    #     """
    #     raise NotImplementedError(
    #         f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
    #     )


def _optical_s_parameter(sax_model: SaxModel):
    class SParameterSax(OpticalSParameter):
    # class SParameterSax():
        optical_ports = sax.get_ports(sax_model)

        def __init__(self, **settings):
            super().__init__(**settings)

        def s_parameters(
            self, 
            wl: ArrayLike, 
            # **kwargs,
        ):
            return sax_model(wl, **self.settings)

    return SParameterSax

COMPONENT_COLOR_DEFAULT = "black"
# COMPONENT_COLOR_SPARAM = "black"
COMPONENT_COLOR_SPARAM = "blue"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"
COMPONENT_COLOR_LOGIC = "gray"


# class ElectricalComponent(Component):
#     pass

# class OpticalComponent(Component):
#     pass

# class LogicComponent(Component):
#     pass


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

    # def _add_models_to_graph(self, models: dict):
    #     for _, node_attr in self.graph.nodes(data=True):
    #         node_attr['model'] = models[node_attr['component']]

    def _convert_sax_models(self):
        for model in self.models:
            component = self.models[model]
            if not inspect.isclass(component):
                s_parameter = _optical_s_parameter(component)
                self.models[model] = s_parameter

    def _mark_component_types(self):
        """ """
        for instance, attr in self.graph.nodes.items():
            model = attr["component"]
            component = self.models[model]

            # if not inspect.isclass(component):
            #     self.graph.nodes[instance]["type"] = "s-parameter: optical"
            #     continue

            tags = set()
            if component.electrical_ports:
                tags.add("electrical")
            if component.logic_ports:
                tags.add("logic")
            if component.optical_ports:
                tags.add("optical")
            # if issubclass(component, ElectricalComponent):
            #     tags.add('electrical')
            # if issubclass(component, LogicComponent):
            #     tags.add('logic')
            # if issubclass(component, OpticalComponent):
            #     tags.add('optical')

            self.graph.nodes[instance]["type"] = "/".join(sorted(tags))
            # if isinstance(component, ElectricalComponent):
            #     tags.add('electrical')
            # if isinstance(component, LogicComponent):
            #     tags.add('logic')
            # if isinstance(component, OpticalComponent):
            #     tags.add('optical')

            # if len(tags) == 0:
            #     self.graph.nodes[instance]['type'] = 's-parameter'
            # else:
            #     self.graph.nodes[instance]['type'] = '/'.join(sorted(tags))

    def _add_ports_to_graph(self):
        for instance, attr in self.graph.nodes.items():
            self.graph.nodes[instance]["electrical ports"] = []
            self.graph.nodes[instance]["logic ports"] = []
            self.graph.nodes[instance]["optical ports"] = []

            model = attr["component"]
            component = self.models[model]

            # if self.graph.nodes[instance]["type"] == "s-parameter: optical":
            #     self.graph.nodes[instance]["optical ports"] = sax.get_ports(component)
            #     self.graph.nodes[instance]["electrical ports"] = []
            #     self.graph.nodes[instance]["logic ports"] = []
            #     continue

            if component.electrical_ports:
                self.graph.nodes[instance]["electrical ports"] = self.models[
                    model
                ].electrical_ports
            if component.logic_ports:
                self.graph.nodes[instance]["logic ports"] = self.models[
                    model
                ].logic_ports
            if component.optical_ports:
                self.graph.nodes[instance]["optical ports"] = self.models[
                    model
                ].optical_ports
            # if issubclass(component, ElectricalComponent):
            #     self.graph.nodes[instance]['electrical ports'] = self.models[model].electrical_ports
            # if issubclass(component, LogicComponent):
            #     self.graph.nodes[instance]['logic ports'] = self.models[model].logic_ports
            # if issubclass(component, OpticalComponent):
            #     self.graph.nodes[instance]['optical ports'] = self.models[model].optical_ports

    def _get_port_type(self, instance, port):
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
            src = edge[0]
            src_port = self.graph.edges[edge]["src_port"]
            src_port_type = self._get_port_type(src, src_port)

            dst = edge[1]
            dst_port = self.graph.edges[edge]["dst_port"]
            dst_port_type = self._get_port_type(dst, dst_port)

            if not src_port_type == dst_port_type:
                raise ValueError("Port types must match")

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
