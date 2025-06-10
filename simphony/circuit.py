from simphony.utils import add_settings_to_netlist, netlist_to_graph
from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent
import gravis as gv
import inspect
import sax

COMPONENT_COLOR_DEFAULT = "black"
COMPONENT_COLOR_SPARAM = "black"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"


class Circuit:
    def __init__(
        self, 
        netlist: dict, 
        models: dict,
        settings: dict = None
    ) -> None:
        if settings is not None:
            add_settings_to_netlist(netlist, settings)
        
        self.netlist = netlist
        self.models = models
        self.settings = settings
        self.graph = netlist_to_graph(netlist)
        
        self._mark_component_types()
        self._add_ports_to_graph()
        self._validate_connections()        
        self._color_nodes()

    def display(self, inline=True):
        fig = gv.d3(self.graph)
        fig.display(inline=inline)
    
    def _add_ports_to_graph(self):
        for instance, attr in self.graph.nodes.items():
            model = attr['component']
            if self.graph.nodes[instance]['type'] == 's-parameter':
                self.graph.nodes[instance]['optical ports'] = sax.get_ports(self.models[model])
                self.graph.nodes[instance]['electrical ports'] = None
                self.graph.nodes[instance]['logic ports'] = None
            else:
                self.graph.nodes[instance]['optical ports'] = self.models[model].optical_ports
                self.graph.nodes[instance]['electrical ports'] = self.models[model].electrical_ports
                self.graph.nodes[instance]['logic ports'] = self.models[model].logic_ports
    def _validate_connections(self):
        pass

    def _mark_component_types(self):
        for instance, attr in self.graph.nodes.items():
            model = attr['component']
            
            if isinstance(self.models[model], OpticalComponent) and isinstance(self.models[model], ElectricalComponent):
                self.graph.nodes[instance]['type'] = 'optoelectrical'
            elif isinstance(self.models[model], OpticalComponent):
                self.graph.nodes[instance]['type'] = 'optical'
            elif isinstance(self.models[model], ElectricalComponent):
                self.graph.nodes[instance]['type'] = 'electrical'
            else:
                self.graph.nodes[instance]['type'] = 's-parameter'
    
    def _color_nodes(self):
        color = COMPONENT_COLOR_DEFAULT
        
        for instance in self.graph.nodes:
            if self.graph.nodes[instance]['type'] == 's-parameter':
                color = COMPONENT_COLOR_SPARAM
            elif self.graph.nodes[instance]['type'] == 'optoelectrical':
                color = COMPONENT_COLOR_OPTOELECTRICAL
            elif self.graph.nodes[instance]['type'] == 'optical':
                color = COMPONENT_COLOR_OPTICAL
            elif self.graph.nodes[instance]['type'] == 'electrical':
                color = COMPONENT_COLOR_ELECTRICAL
            
            self.graph.nodes[instance]['color'] = color