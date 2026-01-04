from simphony.component.component import SampleModeComponent, BlockModeComponent, SParameterComponent
from simphony.component.port import Port

### TODO: Replace this and the future Fork component with VirtualComponent class
# The idea is that in order to prevent these special components from needing to 
# Be defined across simulators, VirtualComponents are special components that the 
# Circuit to InstantiatedFlatDirected circuit knows how to handle
class OpticalDirectionalityTranslator(SampleModeComponent, BlockModeComponent, SParameterComponent):
    """
    It may be natural to link a bidirectional port to an input port of one component
    and an output port of another. One downside of this approach, is that it requires 
    every simphony simulator to update this Component with the appropriate repeator like
    functionality.
    """
    ports = [
        Port(
            name = "bidirectional",
            type = "optical",
            direcitonality = "bidirectional",
        ),
        Port(
            name = "in",
            type = "optical",
            direcitonality = "input",
        ),
        Port(
            name = "out",
            type = "optical",
            direcitonality = "output",
        ),
    ]