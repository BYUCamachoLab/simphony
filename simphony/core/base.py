from importlib import import_module
from typing import List, Dict, Union
import numpy
from simphony.errors import DuplicateModelError

class classproperty(object):
    """Read-only @classproperty decorator.

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

    Examples
    --------
    
    Creation of a cachable component model:

    >>> class RingResonator:
    ...     ports = 4
    ...     s_parameters = s_params
    ...     cachable = True

    Creation of a non-cachable component model:

    >>> def new_s_parameters(self, length, width, thickness):
    ...     # return some calculation based on parameters
    >>> class Waveguide:
    ...     s_parameters = new_s_parameters
    ...     cachable = False
    """

    @classproperty
    def component_type(cls):
        return cls.__name__
        
    ports: int = 0
    s_parameters: Union[list, callable] = None
    cachable: bool = False

    @classmethod
    def get_s_parameters(cls, **kwargs) -> (numpy.array, numpy.array):
        """Returns the s-parameters of the device.

        By default, each ComponentModel takes s_parameters as a keyword 
        argument upon instantiation. When this function is called, it simply 
        returns the frequency and s-parameter matrices that were given. This
        is only true if the model is 'cachable'. If not, then get_s_params can 
        be easily overridden. Simply write a function that returns or 
        calculates the s-parameters for the device (keyword arguments are 
        allowed), and then reassign the `get_s_parameters` reference of the 
        class post-instantiation.

        Parameters
        ----------
        **kwargs
            Derived models may require keyword arguments. See the documentation
            for the models you are using.

        Raises
        ------
        NotImplementedError
            If the ComponentModel is not cachable and 'get_s_parameters' has
            not been overridden.
        """
        if cls.cachable:
            return cls.s_parameters
        else:
            try:
                return cls.s_parameters(**kwargs)
            except:
                print("Class is not cachable and s_parameters is not a function.")


class ComponentInstance():
    """Create an instance in a circuit of an existing ComponentModel.

    An ComponentInstance is a representation of an unique device within a
    circuit or schematic. For example, while a resistor has a ComponentModel,
    R1, R2, and R3 are instances of a resistor.

    Notes
    -----
    Other values can be passed in as "extras". Extras are passed to 
    "get_s_parameters" as keyword arguments. For example, a waveguide needs
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
            The model to be used for this instance.
        nets : list(int), optional
            A list of all port connections (required to be integers). If not
            specified, Netlist will assign them automatically.
        extras : dict, optional
            A dictionary of optional arguments that the model may require for
            calculating s-parameters. For example, length is unique to each
            waveguide and would therefore be considered an 'extra.'
        """
        self.model = model
        self.nets = nets if nets is not None else [None] * model.ports
        self.extras = extras if extras is not None else {}

    def get_s_parameters(self):
        """Get the s-parameters from the linked ComponentModel.
        
        This function simply calls the linked model's 'get_s_parameters()'
        with 'extras' as the keyword arguments."""
        return self.model.get_s_parameters(**self.extras)
