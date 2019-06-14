from simphony import components

# Create your component models here.

class ebeam_wg_integral_1550(components.BaseComponent):
    """This class represents a waveguide component in the netlist. All 
    attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from BaseComponent and inherits all of its data members.

    Attributes
    ----------
    length : float
        Total waveguide length.
    width : float
        Designed waveguide width in microns (um).
    height : float
        Designed waveguide height in microns (um).
    """

    def __init__(self, 
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
        nets : list of ints
            List of connected nets, ordered by port ordering.
        lay_x : float
            The x-position of the component in a layout.
        lay_y : float
            The y-position of the component in a layout.
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
        super().__init__(nets=nets, lay_x=lay_x, lay_y=lay_y)
        self.length = length
        self.width = width
        self.height = height
        self.radius = radius
        self.points = points

    def get_s_parameters(self, length=None, width=None, height=None, delta_length=0):
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
        from simphony.simulation import SimulationSetup as simset
        length = self.length if length is None else length
        width = self.width if width is None else width
        height = self.height if height is None else height
        return self.get_model().get_s_parameters(simset.FREQUENCY_RANGE, length, width, height, delta_length)


    class Metadata:
        simulation_models = [
            ('simphony.elements.ebeam_wg_integral_1550.models', 'ANN_WG', 'Artificial Neural Net'),
            ('simphony.elements.ebeam_wg_integral_1550.models', 'Lumerical_1550', 'Lumerical'),
        ]
        ports = 2