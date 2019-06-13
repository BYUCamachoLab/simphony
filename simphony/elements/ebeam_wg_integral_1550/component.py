from simphony import components

# Create your component models here.

#####

from .models import *

class ebeam_wg_integral_1550(components.BaseComponent):
    """This class represents a waveguide component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from BaseComponent and inherits all of its data members.
    """

    def __init__(self, 
            # component_type: str=self.name
            nets: list=[],
            lay_x: float=0,
            lay_y: float=0,
            length: float=0, 
            width: float=0.5, 
            height: float=0.22, 
            radius: float=0, 
            points: list=[]):
        """Initializes a ebeam_wg_integral_1550 dataclass, which inherits from
        BaseComponent.

        Parameters
        ----------
        length : float
            Total waveguide length.
        width : float
            Designed waveguide width in microns (um).
        height : float
            Designed waveguide height in microns (um).
        radius : float
            The bend radius of waveguide bends.
        points : list of tuples
            A collection of all poitns which define the waveguides' path.
        """
        super().__init__.(*args, **kwargs)
        self.length = length
        self.width = width
        self.height = height
        self.radius = radius
        self.points = points

    def s_parameters(self):
        pass


    class Metadata:
        simulation_models = [
            ('simphony.elements.ebeam_y_1550.models', 'SimModel1'),
            ('simphony.elements.ebeam_y_1550.models', 'SimModel2'),
        ]
        selected_model = 0