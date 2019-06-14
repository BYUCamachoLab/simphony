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

class Component:
    """This class represents an arbitrary type of component, but not an actual
    instance of it.

    Example: A ring resonator is a Component, but your layout may have
    multiple ring resonators on it; those are represented by ComponentInstance.

    This is the abstract base class for all components. All components
    have a type and a list of simulation
    models from which the component in question could retrieve s-parameters.

    Attributes
    ----------
    component_type : str
        The name of the component type. Component types are required to be
        unique; two components with the same type are assumed to be identical
        and have identical properties (width, height, etc.).
    s_parameters : numpy.ndarray
        The s-parameters of the component.

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

    def __init__(self, component_type, s_parameters):
        """Initializes a Component dataclass.

        """
        self.component_type = component_type
        self.s_parameters = s_parameters

    def get_s_parameters(self):
        return s_parameters

    def __str__(self):
        return 'Object::' + str(self.__dict__)



class ComponentInstance():
    """
    Attributes
    ----------
    nets : list of ints
        A list of all connected nets (required to be integers) following the 
        same order that the device's port numbering scheme follows.
    lay_x : float
        The x-position of the component in the overall layout.
    lay_y : float
        The y-position of the component in the overall layout.
    """
    def __init__(self, component: Component=None, nets: list=[], lay_x: float=0, lay_y: float=0):
        """Creates a physical instance of some BaseComponent.

        Parameters
        ----------
        nets : list of ints
            A list of all port connections (required to be integers).
        lay_x : float
            The x-position of the component in the overall layout.
        lay_y : float
            The y-position of the component in the overall layout.
        """
        self.component = component
        self.nets = nets
        self.lay_x = lay_x
        self.lay_y = lay_y



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
