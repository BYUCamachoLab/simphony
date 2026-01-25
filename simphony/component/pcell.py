from simphony.component.component import Component
from simphony.simulation.simulation import SimulationMode
from sax import DEFAULT_MODES

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
        simulation_mode: SimulationMode,
        **kwargs,
    ):
        ...
    
    ### TODO: Make the method here convert to simphony Components firts, instead of instantiate_netlist function
    def _instantiated_netlist(
        self,
        simulation_mode: SimulationMode,
        directed: bool = False,
        default_modes: tuple = DEFAULT_MODES,
    ):
        if self.netlist is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `netlist` before instantiation"
            )

        if self.models is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `models` before instantiation"
            )

        from simphony.circuit.netlist import instantiate_netlist

        return instantiate_netlist(
            self.netlist,
            self.models,
            self.settings,
            simulation_mode,
            directed=directed,
            default_modes=default_modes,
        )
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