from simphony.component.component import BlockModeComponent, SampleModeComponent, Component
from simphony.component.port import Port
from sax import DEFAULT_MODES

class ModeConverter(
    SampleModeComponent,
    BlockModeComponent,
):
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

def mode_multiplexer(
    input_modes: tuple|list = DEFAULT_MODES
) -> type[Component]:
    input_port_names = [f"{mode}" for mode in input_modes]

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
                name='out',
                type="optical",
                directionality="output"
            )
        ]

    return ModeMultiplexer

def mode_demultiplexer(
    output_modes: tuple|list = DEFAULT_MODES
) -> type[Component]:
    output_port_names = [f"{mode}" for mode in output_modes]

    class ModeDemultiplexer(
        SampleModeComponent, 
        BlockModeComponent
    ):
        ports = [
            Port(
                name='in',
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
    
    return ModeDemultiplexer

