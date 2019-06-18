from importlib import import_module
from typing import List, Dict
import numpy
from simphony.errors import DuplicateModelError

def clear_models():
    """Clears all models loaded in program memory."""
    ComponentModel.clear_models()

class ComponentModel:
    """The base class for all component models.
    
    This class represents the model for some arbitrary type of component, 
    but not an actual instance of it. For example, a ring resonator is a 
    ComponentModel, but your layout may have multiple ring resonators on it; 
    those are represented by ComponentInstance.

    Examples
    --------
    
    Creation of a cachable component model:

    >>> rr = simphony.core.ComponentModel("ring_resonator", s_params, cachable=True)

    Creation of a non-cachable component model:

    >>> wg = simphony.core.ComponentModel("waveguide", s_params, cachable=False)
    >>> def new_s_parameters(length, width, thickness):
    ...     # return some calculation based on parameters
    >>> wg.get_s_parameters = new_s_parameters
    """
    models = {}

    @classmethod
    def clear_models(cls):
        cls.models = {}

    def __init__(self, component_type, s_parameters=None, cachable=False):
        """Initializes a ComponentModel dataclass.

        A ComponentModel represents a type of component or device within a 
        circuit. It is not an instance of that device, however; for example,
        an electrical circuit can be constructed from resistors, transistors,
        diodes, etc. But specific resistors, transistors, and diodes, and 
        their locations or connections are specified as a ComponentInstance.

        Parameters
        ----------
        component_type : str
            A unique name specifying the type of this component.
        s_parameters : numpy.array
            A tuple, '(f,s)', where 'f' is the frequency array corresponding to
            's', a matrix containing the s-parameters of the device.
        cachable : bool
            True if the s-parameters are static; false if they depend on other
            variables.

        Raises
        ------
        ValueError
            If cachable=True and s_parameters are not specified.

        See Also
        --------
        ComponentInstance : Component instances that reference ComponentModel.
        """
        if component_type in self.models.keys(): #self._component_types:
            raise DuplicateModelError(component_type)
        else:
            self._component_type = component_type
            # self._component_types.add(component_type)

        if cachable:
            if s_parameters is None:
                raise ValueError("\'s_parameters\' cannot be None if cachable=True.")
            self.s_parameters = s_parameters
        self.cachable = cachable
        self.models[component_type] = self

    @property
    def component_type(self):
        return self._component_type

    @component_type.setter
    def component_type(self, value):
        if(value in self.models.keys()):#self._component_types):
            raise DuplicateModelError(value)
        self._component_type = value

    def get_s_parameters(self, **kwargs) -> (numpy.array, numpy.array):
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
            If the ComponentModel is not cachable.
        """
        if self.cachable:
            return self.s_parameters
        else:
            raise NotImplementedError("Component s-parameters are not cachable and 'get_s_parameters()' is not defined.")

    def __copy__(self):
        raise DuplicateModelError(self.component_type)

    def __deepcopy__(self, memo):
        raise DuplicateModelError(self.component_type)


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
    height : float
        Designed waveguide height in microns (um).
    radius : float
        The bend radius of waveguide bends.
    points : list of tuples
        A collection of all points which define the waveguides' path.
    """
    def __init__(self, model: ComponentModel=None, nets: List=None, lay_x: float=0, lay_y: float=0, extras: Dict=None):
        """Creates an instance of some ComponentModel.

        Parameters
        ----------
        nets : list(int)
            A list of all port connections (required to be integers).
        lay_x : float
            The x-position of the component in the overall layout.
        lay_y : float
            The y-position of the component in the overall layout.
        """
        self.model = model
        self.nets = nets if nets is not None else []
        self.lay_x = lay_x
        self.lay_y = lay_y
        self.extras = extras if extras is not None else {}

    def get_s_parameters(self):
        """Get the s-parameters from the linked ComponentModel."""
        return self.model.get_s_parameters(**self.extras)
