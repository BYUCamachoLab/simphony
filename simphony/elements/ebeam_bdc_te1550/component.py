from simphony import components

# Create your component models here.

class ebeam_bdc_te1550(components.BaseComponent):
    """This class represents a bidirectional coupler component in the netlist.
    All attributes can be initialized as keyword arguments in the __init__ 
    function.

    This class inherits from BaseComponent and inherits all of its data members.

    Attributes
    ----------
    Inherited
    """
    def __init__(self, nets: list=[], lay_x: float=0, lay_y: float=0):
        """Initializes a ebeam_bdc_te1550 dataclass, which inherits from 
        BaseComponent.

        Parameters
        ----------
        nets : list of ints
            List of connected nets, ordered by port ordering.
        lay_x : float
            The x-position of the component in a layout.
        lay_y : float
            The y-position of the component in a layout.
        """
        super().__init__(nets=nets, lay_x=lay_x, lay_y=lay_y)

    def get_s_parameters(self):
        print(self.get_model())
        # freq, sparams = self._model_ref.get_s_params(self.port_count)
        # return simset.interpolate(freq, sparams)

    class Metadata:
        simulation_models = [
            ('simphony.elements.ebeam_bdc_te1550.models', 'BDC_Model'),
        ]
        ports = 4