from simphony.component.component import Component
from simphony.simulation.simulation import SimulationParameters
# from sax import DEFAULT_MODES
import inspect
import sax
from copy import deepcopy

### TODO: Decide whether PCells are Components or not, gotta love OOP
class PCell(Component):
    """
    Some circuit components will not have a well defined, internal structure 
    until after they are parameterized. For this case, we use parameterized 
    cells, or PCells. 

    The Circuit objects will display the PCell as a single component. 
    
    All PCell classes must have well defined external ports, prior to instantiation.

    Similar to the Component Classes, the PCell class is a baseclass for special 
    Component objects that require a dynamic structure at the simulation runtime.
    Therefore, the __init__() function should be called by the InstantiatedCircuit class, and 
    not the designer of the PCell. 

    The PCell designer should simply inherit from the PCell baseclass and overwrite the 
    appropriate class fields and methods. 

    Models may be modified or left alone before the conclusion of 
    the __init__() function
    """
    ports = None
    
    netlist = None
    models = None
    settings = None

    def __repr__(self):
        return f"<{type(self).__name__} (PCell obj)>"

    def __init__(
        self,
        simulation_mode: SimulationParameters,
        **kwargs,
    ):
        ...
    
    ### TODO: Make the method here convert to simphony Components firts, instead of instantiate_netlist function
    def _instantiated_netlist(
        self,
        simulation_parameters: SimulationParameters,
        # directed: bool = False,
        # default_modes: tuple = DEFAULT_MODES,
    ):
        """
        If directed is true, then 
        """
        if self.netlist is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `netlist` before instantiation"
            )

        if self.models is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `models` before instantiation"
            )

        from simphony.circuit.netlist import instantiate_netlist

        port_directionality = {port.name:port.directionality for port in self.ports}

        new_netlist, new_models, new_settings = _convert_sax_models(
            simulation_parameters,
            self.netlist, 
            self.models, 
            self.settings,
            # directed, 
            # default_modes, 
            # port_directionality
            )

        instantiated_netlist = instantiate_netlist(
            new_netlist,
            new_models,
            new_settings,
            simulation_parameters,
            # directed=directed,
            # default_modes=default_modes,
        )

        # self.external_port_aliases = instantiated_netlist['ports']

        return instantiated_netlist
    # def _instantiated_netlist(
    #     self,
    #     directed: bool = False,
    #     default_modes: tuple = DEFAULT_MODES,
    #     settings: dict = None,
    #     ):
    #     from simphony.circuit.netlist import instantiate_netlist

    #     return instantiate_netlist(
    #         self.netlist, 
    #         self.models,
    #         directed = directed,
    #         default_modes = default_modes,
    #         settings = settings
    #     )

def _convert_sax_models(
    simulation_parameters,
    netlist, 
    models,  
    settings,
    # directed,
    # default_modes,
    # port_directionality,    
):
    from simphony.circuit.netlist import add_settings_to_netlist
    from simphony.libraries.ideal.s_parameters import optical_s_parameter
    
    netlist = sax.netlist(deepcopy(netlist))
    for _, subnetlist in netlist.items():
        add_settings_to_netlist(subnetlist) # Just to normalize, we will use the settings the user provided later
    netlist = sax.flatten_netlist(sax.netlist(netlist))
    new_netlist = deepcopy(netlist)
    new_settings = deepcopy(settings)
    # new_netlist = netlist
    new_models = {}
    unique_str = "_X_"
    for key in netlist['instances'].keys():
        while unique_str in key:
            unique_str += "_"
    unique_word = unique_str + "DIRECTED_SPARAMETER_MODEL"
    count = 0

    external_port_lut = {v: k for k, v in netlist['ports'].items()}
    for instance_name, instance_data in netlist['instances'].items():
        model_name = instance_data['component']
        model = models[model_name]
        if inspect.isclass(model):
            new_models[model_name] = model
            continue

        if not simulation_parameters.directed:
            ### TODO: Make the external port directionality match the pcell statement
            ### "directed" only decides whether internal connection directionality should be interpretted based
            ### on order, but the specification in the ports list can trump this
            # print(port_directionality)
            directionality = None # Defaults to bidirectional
            new_models[model_name] = optical_s_parameter(model, directionality, simulation_parameters.mode_identifiers)
            continue
        
        raise ValueError("Directed Simulations Require Component Types, no sax models allowed. This is because sax models are bidirectional")
        # count += 1
        # ### If directed, then we will have to make a new model for each s-parameter element
        # directionality = {
        #     port_name : (
        #         "output" if instance_name + "," + port_name in new_netlist['connections'].keys()
        #         else "input" if instance_name + "," + port_name in new_netlist['connections'].values()
        #         else None if False
        #         else "bidirectional"
        #     )
        #     for port_name in sax.get_ports(model())
        # }

        # new_model_name = unique_word + f"_{count}_" + model_name
        # new_models[new_model_name] = optical_s_parameter(model, directionality, default_modes)
        pass
        
        # if issubclass(model, Component) or issubclass(model, PCell):
        #     new_models[model_name] = model
        # elif not directed:
        #     directionality = "bidirectional"
        #     model = optical_s_parameter(model, directionality, default_modes)
        # elif directed:
        #     ### TODO: Find Directionality
        #     directionality = ""
        #     model = optical_s_parameter(model, directionality, default_modes)
    
    new_settings
    
    # TODO: Remove duplicate code. This is taken from InstantiatedCircuit __init__
    from simphony.libraries.ideal.s_parameters import SParameterPlaceholder
    # Reinterpret Sax Settings to optical_s_parameter Component settings
    # for instance_name, instance_settings in settings.items():
    for instance_name in new_netlist['instances'].keys():
        model_name = new_netlist['instances'][instance_name]['component']
        if issubclass(new_models[model_name], SParameterPlaceholder) and not "sax_settings" in settings[instance_name].keys():
            new_settings[instance_name] = {"sax_settings": settings[instance_name]}

    return new_netlist, new_models, new_settings