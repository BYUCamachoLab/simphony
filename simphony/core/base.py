from importlib import import_module
from typing import Dict, List, Union

import numpy
from simphony.errors import DuplicateModelError


class classproperty(object):
    """Read-only @classproperty decorator. Allows using class names for models
    as a property.

    Solution from https://stackoverflow.com/questions/128573/using-property-on-classmethods/13624858#13624858
    """
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

class ComponentModel:
    """The base class for all component models.
    
    This class represents the model for some arbitrary type of component, 
    but not an actual instance of it. For example, a ring resonator is a 
    ComponentModel, but your layout may have multiple ring resonators on it; 
    those are represented by ComponentInstance.

    Attributes
    ----------
    component_type : str
        A read-only property that returns the class name of the model (class 
        names should be descriptive of the model).
    ports : int
        The number of ports this component has.
    s_parameters : tuple, callable
        If cachable, s_parameters is a tuple of the form 
        ([frequency array], [s-parameter matrix]). If not cachable, the
        s-parameters must depend on some number of arguments; hence,
        s-parameters should be reassigned to some function that can take any
        number of arguments.
    cachable : bool
        A boolean value. If cachable, s-parameters will only be calculated 
        once during simulation, and reused for each component instance. If 
        not cachable, the parameters stored within each component instance 
        will be used to recalculate the s-parameters each time using the
        now-defined `s_parameters()` function.

    Examples
    --------
    
    Creation of a cachable component model:

    >>> @register_component_model
    ... class RingResonator(ComponentModel):
    ...     ports = 4
    ...     s_parameters = s_params
    ...     cachable = True

    Creation of a non-cachable component model:

    >>> def new_s_parameters(self, length, width, thickness, **kwargs):
    ...     # return some calculation based on parameters
    >>> @register_component_model
    ... class Waveguide(ComponentModel):
    ...     ports = 2
    ...     s_parameters = new_s_parameters
    ...     cachable = False

    >>> @register_component_model
    ... class Waveguide(ComponentModel):
    ...     ports = 2
    ...     def s_parameters(self, length, width, thickness, **kwargs):
    ...         # return some calculation based on parameters
    ...     cachable = False
    """

    @classproperty
    def component_type(cls):
        """Returns the name of the class, which is the type of the component.
        """
        return cls.__name__
        
    ports: int = 0
    s_parameters: Union[tuple, callable] = None
    cachable: bool = False

    @classmethod
    def get_s_parameters(cls, **kwargs) -> (numpy.array, numpy.array):
        """Returns the s-parameters of the device.

        Usually, each ComponentModel has an s_parameters attribute when 
        the class is defined. When `get_s_parameters` is called, it simply 
        returns the frequency and s-parameter matrices that were defined. This
        is only true if the model is 'cachable'. If not, then `s_parameters` 
        should be overridden. Simply write a function that returns or 
        calculates the s-parameters for the device (keyword arguments are 
        allowed), and then reassign the `s_parameters` reference of the 
        class. `get_s_parameters()` will then call that function with 
        the parameters available in each `ComponentInstance` object.

        Parameters
        ----------
        **kwargs
            Derived models may require keyword arguments. See the documentation
            for the models you are using.

        Raises
        ------
        NotImplementedError
            If the ComponentModel is not cachable and 's_parameters' has
            not been assigned as a function.
        """
        if cls.cachable:
            return cls.s_parameters
        else:
            try:
                return cls.s_parameters(**kwargs)
            except:
                raise NotImplementedError("Class is not cachable and s_parameters is not a function.")


class ComponentInstance:
    """Create an instance in a circuit of an existing ComponentModel.

    An ComponentInstance is a representation of an unique instance of a device 
    within a circuit or schematic. For example, while a resistor has a 
    ComponentModel, R1, R2, and R3 are instances of a resistor.

    Notes
    -----
    Other values can be passed in as "extras". Extras are passed to the model's
    `get_s_parameters()` as keyword arguments. For example, a waveguide needs
    these values to properly calculate its s-parameters:
    length : float
        Total waveguide length.
    width : float
        Designed waveguide width in microns (um).
    thickness : float
        Designed waveguide thickness in microns (um).
    radius : float
        The bend radius of waveguide bends.
    """
    def __init__(self, model: ComponentModel=None, nets: List=None, extras: Dict=None):
        """Creates an instance of some ComponentModel.

        Parameters
        ----------
        model: ComponentModel
            The model this instance is based on.
        nets : list(int), optional
            A list of all port connections (required to be integers). If not
            specified, Netlist will assign them automatically.
        extras : dict, optional
            A dictionary of optional arguments (and their values) that the 
            model may require for calculating s-parameters. For example, 
            length is unique to each waveguide and would therefore be 
            considered a necessary 'extra' to simulate each instance.
            `extras` can always be updated/added to post-instantiation.
        """
        self.model = model
        self.nets = nets if nets is not None else [None] * model.ports
        self.extras = extras if extras is not None else {}

    def get_s_parameters(self):
        """Get the s-parameters from the linked ComponentModel.
        
        This function simply calls the linked model's 'get_s_parameters()'
        with 'extras' as the keyword arguments."""
        return self.model.get_s_parameters(**self.extras)
