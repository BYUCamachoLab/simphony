"""
components.py

Author: Sequoia Ploeg

Dependencies:
- sys
    Required for building classes from various modules using their string
    classnames.
- importlib
    Dynamically imports the installed component models.

This file dynamically loads all installed components within the models package.
It also provides object models for netlist capabilities (all components 
can be formatted as JSON).
"""


"""
This is where you should list all installed components from which you plan to 
get s-parameters. In addition to listing the modules here, make sure to list 
the relevant modules within each component class below, too, under 
_simulation_model.
"""
INSTALLED_COMPONENTS = [
    'wg_ann',
    'wg1550_lumerical',
    'ebeam_bdc_te1550',
    'ebeam_y_1550',
    'ebeam_dc_halfring_te1550',
    'ebeam_terminator_te1550',
    'ebeam_gc_te1550',
]

"""
BEGIN DO NOT ALTER
"""
import sys
from importlib import import_module

LOADED_MODELS = {}

for component in INSTALLED_COMPONENTS:
    mod = import_module('.' + component, __name__.split('.')[0] + '.models')
    LOADED_MODELS[component] = mod.Model

from abc import ABC, abstractmethod
from ..simulation import SimulationSetup as simset
"""
END DO NOT ALTER
"""

class Component(ABC):
    """This class represents an arbitrary component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This is the abstract base class for all netlist components. All components
    have a type, a list of nets, x/y layout positions, and a list of simulation
    models from which the component in question could retrieve s-parameters.

    All class extensions should provide additional data members with type hints
    and default values if necessary.

    Class Attributes
    ----------------
    _simulation_models : dict
        A dictionary of installed models (packages) from which this component 
        could retrieve s-parameters. This is a class attribute, and is not 
        stored at the instance level. It's format ought to be as follows:
        {'[Human Readable Name]': '[Module Name]'}.
    _selected_model : str
        A key from the _simulation_models dictionary.
    _model_ref : class
        A reference to the Model class of the currently selected model for this
        component type. Provides a hook for calling functions like 
        'get_s_params'.

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
    component_type: str = None
    nets: list = []
    lay_x: float = 0
    lay_y: float = 0
    _simulation_models: dict = {}
    _selected_model: str

    @staticmethod
    def setup(obj):
        obj._selected_model = next(iter(obj._simulation_models)) if obj._simulation_models else None
        obj._model_ref = LOADED_MODELS[obj._simulation_models[obj._selected_model]] if obj._selected_model else None

    def __init__(self, *args, **kwargs):
        """Initializes a Component dataclass.

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
        if 'nets' in kwargs:
            self.nets = kwargs.get('nets')
        if 'lay_x' in kwargs:
            self.lay_x = kwargs.get('lay_x')
        if 'lay_y' in kwargs:
            self.lay_y = kwargs.get('lay_y')

    @classmethod
    def set_model(cls, key):
        cls._selected_model = key
        cls._model_ref = LOADED_MODELS[cls._simulation_models[cls._selected_model]] if cls._selected_model else None
        cls._model_ref.about()

    @property
    def port_count(self):
        return len(self.nets)

    @abstractmethod
    def get_s_params(self):
        freq, sparams = self._model_ref.get_s_params(self.port_count)
        return simset.interpolate(freq, sparams)

    def __str__(self):
        return 'Object::' + str(self.__dict__)




class ebeam_wg_integral_1550(Component):
    """This class represents a waveguide component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    length : float
        Total waveguide length.
    width : float
        Designed waveguide width in microns (um).
    height : float
        Designed waveguide height in microns (um).
    """
    length: float = 0
    width: float = 0.5
    height: float = 0.22
    radius: float = 0
    points: list = []

    _simulation_models = {
        'ANN Waveguide': 'wg_ann',
        'EBeam Waveguide': 'wg1550_lumerical',
    }

    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_wg_integral_1550 dataclass, which inherits from
        Component.

        Parameters
        ----------
        length : float
            Total waveguide length.
        width : float
            Designed waveguide width.
        height : float
            Designed waveguide height.
        points : list of tuples
        """
        super().__init__(*args, **kwargs)
        if 'length' in kwargs:
            self.length = kwargs.get('length')
        if 'width' in kwargs:
            self.width = kwargs.get('width')
        if 'height' in kwargs:
            self.height = kwargs.get('height')
        if 'points' in kwargs:
            self.points = kwargs.get('points')

    def get_s_params(self, length=None, width=None, height=None, delta_length=0):
        """
        Gets the s-parameter matrix for this component.

        Parameters
        ----------
        length : float, optional
            Length of the waveguide.
        width : float, optional
            Width of the waveguide in microns (um).
        height : float, optional
            Height of the waveguide in microns (um).
        delta_length :  : float, optional
            Percentage difference in the length of the waveguide as a float 
            (e.g. '0.1' -> 10%).

        Returns
        -------
        (np.array, np.array)
            A tuple; the first value is the frequency range, the second value 
            is its corresponding s-parameter matrix.
        """
        length = self.length if length is None else length
        width = self.width if width is None else width
        height = self.height if height is None else height
        return self._model_ref.get_s_params(simset.FREQUENCY_RANGE, length, width, height, delta_length)




class ebeam_bdc_te1550(Component):
    """This class represents a bidirectional coupler component in the netlist.
    All attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    _simulation_models = {
        'EBeam BDC': 'ebeam_bdc_te1550',
    }
    
    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_bdc_te1550 dataclass, which inherits from 
        Component.

        Parameters
        ----------
        Inherited
        """
        super().__init__(*args, **kwargs)

    def get_s_params(self):
        return super().get_s_params()




class ebeam_gc_te1550(Component):
    """This class represents a grating coupler component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    _simulation_models = {
        'EBeam Grating Coupler': 'ebeam_gc_te1550',
    }
    
    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_gc_te1550 dataclass, which inherits from 
        Component.

        Parameters
        ----------
        Inherited
        """
        super().__init__(*args, **kwargs)

    def get_s_params(self):
        return super().get_s_params()




class ebeam_y_1550(Component):
    """This class represents a Y-branch component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    _simulation_models = {
        'EBeam Y-Branch': 'ebeam_y_1550',
    }
    
    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_y_1550 dataclass, which inherits from Component.

        Parameters
        ----------
        Inherited
        """
        super().__init__(*args, **kwargs)
    
    def get_s_params(self):
        return super().get_s_params()




class ebeam_terminator_te1550(Component):
    """This class represents a terminator component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    _simulation_models = {
        'EBeam Nanotaper Terminator': 'ebeam_terminator_te1550',
    }
    
    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_terminator_te1550 dataclass, which inherits from
        Component.

        Parameters
        ----------
        Inherited
        """
        super().__init__(*args, **kwargs)

    def get_s_params(self):
        return super().get_s_params()




class ebeam_dc_halfring_te1550(Component):
    """This class represents a half-ring component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from Component and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    _simulation_models = {
        'EBeam DC Halfring': 'ebeam_dc_halfring_te1550',
    }
    
    def __init__(self, *args, **kwargs):
        """Initializes a ebeam_dc_halfring_te1550 dataclass, which inherits 
        from Component.

        Parameters
        ----------
        Inherited
        """
        super().__init__(*args, **kwargs)

    def get_s_params(self):
        return super().get_s_params()




"""
BEGIN DO NOT ALTER
"""
# Finish setting all class variables for component subclasses
comp_subclasses = [class_ for class_ in Component.__subclasses__()]
for class_ in comp_subclasses:
    class_.setup(class_)

def create_component_by_name(component_name: str):
    return getattr(sys.modules[__name__], component_name)()
"""
END DO NOT ALTER
"""