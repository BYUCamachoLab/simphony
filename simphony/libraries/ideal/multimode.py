from simphony.component.component import BlockModeComponent, SampleModeComponent, Component
from simphony.component.port import Port
from sax import DEFAULT_MODES

### TODO: Implement ModeConvert
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
    
    def __init__(**kwargs):
        pass

### TODO: Implement ModeMultiplexer
def mode_multiplexer(
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

        def __init__(**kwargs):
            pass

    return ModeMultiplexer

### TODO: Implement ModeDemultiplexer
def mode_demultiplexer(
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

        def __init__(**kwargs):
            pass
        
    return ModeDemultiplexer

