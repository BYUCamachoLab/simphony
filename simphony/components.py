"""
components.py

Author: Sequoia Ploeg

Dependencies:
- importlib
    Dynamically imports the installed component models.

This file dynamically loads all installed components within the models package.
It also provides object models for netlist capabilities (all components 
can be formatted as JSON).
"""

from importlib import import_module
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """This class represents an arbitrary component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This is the abstract base class for all netlist components. All components
    have a type, a list of nets, x/y layout positions, and a list of simulation
    models from which the component in question could retrieve s-parameters.

    All class extensions should provide additional data members with type hints
    and default values if necessary.

    Attributes
    ----------
    component_type : str
        The name of the component type.
    nets : list of ints
        A list of all connected nets (required to be integers) following the 
        same order that the device's port numbering scheme follows.
    lay_x : float
        The x-position of the component in the overall layout.
    lay_y : float
        The y-position of the component in the overall layout.
    port_count : int
        The number of ports on the device, calculated by counting the number of
        connected nets. This attribute is a property and is not settable.

    Methods
    -------
    set_model(key: str)
        Class method for selecting a simulation model for the specified 
        component.
    get_s_params(*args, **kwargs) : np.array, np.array
        Abstract method that each class implements. Each classes passes in the
        necessary parameters to its model and returns the frequency and 
        s-parameter matrices.
    """

    def __init__(self, nets: list=[], lay_x: float=0, lay_y: float=0):
        """Initializes a BaseComponent dataclass.

        Parameters
        ----------
        nets : list of ints
            A list of all port connections (required to be integers).
        lay_x : float
            The x-position of the component in the overall layout.
        lay_y : float
            The y-position of the component in the overall layout.
        """
        self.component_type = type(self).__name__
        self.selected_model = 0
        self.nets = nets
        self.lay_x = lay_x
        self.lay_y = lay_y


    @property
    def port_count(self):
        return self.Metadata.ports

    @abstractmethod
    def get_s_parameters(self):
        raise NotImplementedError

    def __str__(self):
        return 'Object::' + str(self.__dict__)

    def register_model(module, model):
        pass

    def get_model(self):
        selected_module = self.Metadata.simulation_models[self.selected_model]
        mod = import_module(selected_module[0])
        return getattr(mod, selected_module[1])

    @classmethod
    def get_all_models(cls):
        models = {}
        for model in cls.Metadata.simulation_models:
            module_name, class_name, human_readable_name = model
            mod = import_module(module_name)
            models[human_readable_name] = getattr(mod, class_name)
        return models




class SimulationModel(ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_s_parameters(numports: int):
        """Returns the s-parameters across some frequency range for the ebeam_dc_halfring_te1550 model
        in the form [frequency, s-parameters].

        Parameters
        ----------
        numports : int
            The number of ports the photonic component has.
        """
        pass


from importlib import import_module
import pkgutil, inspect
import simphony.elements

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages.

    Courtesy of https://stackoverflow.com/a/25562415/11530613

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

def load_elements():
    # Courtesy of https://stackoverflow.com/a/15833780/11530613
    mods = inspect.getmembers(simphony.elements, inspect.ismodule)
    for mod in mods:
        # print(mod)
        import_submodules(mod[1])

load_elements()

LOADED_COMPONENTS = {}
for c in BaseComponent.__subclasses__():
    LOADED_COMPONENTS[c.__name__] = c

def get_all_components():
    return BaseComponent.__subclasses__()

def get_all_models():
    return SimulationModel.__subclasses__()

def component_verifier(component):
    if len(component.Metadata.simulation_models) == 0:
        raise ImportError(type(component).__name__ + " has no simulation models defined.")

def create_component_by_name(component_name: str):
    return LOADED_COMPONENTS[component_name]()
