from simphony.component.component import BlockModeComponent, SampleModeComponent, Component
from simphony.component.port import Port
from sax import DEFAULT_MODES
from simphony.simulation.simulation import SimulationParameters

### TODO: Implement ModeConvert
class ModeConverter(
    SampleModeComponent,
    BlockModeComponent,
):
    """
    Takes in a signal on a single mode (any mode), and outputs a signal on a single, specified mode
    
    Will ignore all modes except for the first in the signal passed to it
    """
    ports = [
            Port(
                name='in',
                type="optical",
                directionality="input"
            ) 
        ] + [
            Port(
                name='out', 
                type="optical", 
                directionality = 'output'
            )
        ]
    
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        output_mode="TE",
    ):
        self.simulation_parameters = simulation_parameters
        self.output_mode = output_mode

    def block_mode_response(self, inputs, simulation_parameters):
        # TODO: IMPLEMENT MODE CONVERTER
        pass
        return ...

### TODO: Implement ModeMultiplexer
def mode_multiplexer(
    simulation_parameters: SimulationParameters,
    *,
    input_modes: tuple|list = DEFAULT_MODES,
    output_port_name: str = "out_port",
    input_port_suffix: str = "_port",

) -> type[Component]:
    input_port_names = [f"{mode}{input_port_suffix}" for mode in input_modes]

    class ModeMultiplexer(
        SampleModeComponent, 
        BlockModeComponent
    ):
        ports = [
            Port(
                name=port_name, 
                type="optical", 
                directionality = 'input'
            ) 
            for port_name in input_port_names
        ] + [
            Port(
                name=output_port_name,
                type="optical",
                directionality="output"
            )
        ]

        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            **kwargs,
        ):
            self.simulation_parameters = simulation_parameters
    
        def block_mode_response(self, inputs, simulation_parameters):
            # TODO: IMPLEMENT ME
            pass
            return ...


    return ModeMultiplexer



### TODO: Implement ModeDemultiplexer
def mode_demultiplexer(
    simulation_parameters: SimulationParameters,
    *,
    output_modes: tuple|list = DEFAULT_MODES,
    input_port_name: str = "in_port",
    output_port_suffix: str = "_port",
) -> type[Component]:
    output_port_names = [f"{mode}{output_port_suffix}" for mode in output_modes]

    class ModeDemultiplexer(
        SampleModeComponent, 
        BlockModeComponent
    ):
        ports = [
            Port(
                name=input_port_name,
                type="optical",
                directionality="input"
            ) 
        ] + [
            Port(
                name=port_name, 
                type="optical", 
                directionality = 'output'
            ) 
            for port_name in output_port_names
        ]

        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            **kwargs,
        ):
            pass
    
        def block_mode_response(self, inputs, simulation_parameters):
            # TODO: IMPLEMENT ME
            pass
            return ...
    
    return ModeDemultiplexer

