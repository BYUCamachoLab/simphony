from simphony.utils import add_settings_to_netlist, netlist_to_graph
from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
import gravis as gv
import inspect
import sax

COMPONENT_COLOR_DEFAULT = "black"
# COMPONENT_COLOR_SPARAM = "black"
COMPONENT_COLOR_SPARAM = "blue"
COMPONENT_COLOR_OPTICAL = "blue"
COMPONENT_COLOR_ELECTRICAL = "red"
COMPONENT_COLOR_OPTOELECTRICAL = "purple"
COMPONENT_COLOR_LOGIC = "gray"


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

    def _mark_component_types(self):
        """
        """
        for instance, attr in self.graph.nodes.items():
            model = attr['component']
            component = self.models[model]
            tags = set()
            
            if isinstance(component, ElectricalComponent):
                tags.add('electrical')
            if isinstance(component, LogicComponent):
                tags.add('logic')
            if isinstance(component, OpticalComponent):
                tags.add('optical')
            
            
            if len(tags) == 0:
                self.graph.nodes[instance]['type'] = 's-parameter'
            else:
                self.graph.nodes[instance]['type'] = '/'.join(sorted(tags))


            
            # if isinstance(self.models[model], OpticalComponent) and isinstance(self.models[model], ElectricalComponent):
            #     self.graph.nodes[instance]['type'] = 'electrical/optical'
            # elif isinstance(self.models[model], OpticalComponent):
            #     self.graph.nodes[instance]['type'] = 'optical'
            # elif isinstance(self.models[model], ElectricalComponent):
            #     self.graph.nodes[instance]['type'] = 'electrical'
            # else:
            #     self.graph.nodes[instance]['type'] = 's-parameter'
     
    def _add_ports_to_graph(self):
        for instance, attr in self.graph.nodes.items():
            model = attr['component']
            if self.graph.nodes[instance]['type'] == 's-parameter':
                self.graph.nodes[instance]['optical ports'] = sax.get_ports(self.models[model])
                self.graph.nodes[instance]['electrical ports'] = []
                self.graph.nodes[instance]['logic ports'] = []
            else:
                self.graph.nodes[instance]['optical ports'] = self.models[model].optical_ports
                self.graph.nodes[instance]['electrical ports'] = self.models[model].electrical_ports
                self.graph.nodes[instance]['logic ports'] = self.models[model].logic_ports
    
    def _get_port_type(self, instance, port):
        optical_ports = self.graph.nodes[instance]['optical ports']
        electrical_ports = self.graph.nodes[instance]['electrical ports']
        logic_ports = self.graph.nodes[instance]['logic ports']

        if port in optical_ports:
            return 'optical'
        elif port in electrical_ports:
            return 'electrical'
        elif port in logic_ports:
            return 'logic'

    def _validate_connections(self):
        # Unique port names assumed (enforced in TimeSystem)
        
        # Verify optical-to-optical, electrical-to-electrical, logic-to-logic
        for edge in self.graph.edges:
            src = edge[0]
            src_port = self.graph.edges[edge]['src_port']
            src_port_type = self._get_port_type(src, src_port)

            dst = edge[1]
            dst_port = self.graph.edges[edge]['dst_port']
            dst_port_type = self._get_port_type(dst, dst_port)

            if not src_port_type == dst_port_type:
                raise ValueError("Port types must match")

    def _color_nodes(self):
        color = COMPONENT_COLOR_DEFAULT
        
        # Assumes tags in alphabetical order 
        for instance in self.graph.nodes:
            if self.graph.nodes[instance]['type'] == 's-parameter':
                color = COMPONENT_COLOR_SPARAM
            elif self.graph.nodes[instance]['type'] == 'electrical/optical':
                color = COMPONENT_COLOR_OPTOELECTRICAL
            elif self.graph.nodes[instance]['type'] == 'optical':
                color = COMPONENT_COLOR_OPTICAL
            elif self.graph.nodes[instance]['type'] == 'electrical':
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]['type'] == 'logic':
                color = COMPONENT_COLOR_LOGIC
            elif self.graph.nodes[instance]['type'] == 'electrical/logic':
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]['type'] == 'logic/optical':
                color = COMPONENT_COLOR_ELECTRICAL
            elif self.graph.nodes[instance]['type'] == 'electrical/logic/optical':
                color = COMPONENT_COLOR_OPTOELECTRICAL
            
            self.graph.nodes[instance]['color'] = color