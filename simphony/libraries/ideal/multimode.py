from simphony.component.component import BlockModeComponent, SampleModeComponent, Component
from simphony.signal.block_mode import BlockModeOpticalSignal
from simphony.component.port import Port
from sax import DEFAULT_MODES
from simphony.simulation.simulation import SimulationParameters
import jax.numpy as jnp

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
        input_mode="TE",
        output_mode="TM",
    ):
        self.simulation_parameters = simulation_parameters
        self.input_mode = input_mode
        self.output_mode = output_mode

    def block_mode_response(self, inputs, simulation_parameters):
        # TODO: IMPLEMENT MODE CONVERTER
        input_mode_index = simulation_parameters.mode_identifiers.index(self.input_mode)
        output_mode_index = simulation_parameters.mode_identifiers.index(self.output_mode)
        
        input_amplitude = inputs['in'].amplitude
        wavelength = inputs['in'].wavelength
        
        output_amplitude = jnp.zeros_like(input_amplitude)
        output_amplitude = output_amplitude.at[:, :, output_mode_index].set(input_amplitude[:, :, input_mode_index])
        
        outputs = {
            'out': BlockModeOpticalSignal(amplitude=output_amplitude, wavelength=wavelength),
        }

        return outputs

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

        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            **kwargs,
        ):
            self.simulation_parameters = simulation_parameters
            self.input_port_names = input_port_names
            self.output_port_name = output_port_name
    
        def block_mode_response(self, input_signals, simulation_parameters):
            outputs = {}
            _input_amplitude = list(input_signals.values())[0].amplitude
            wavelengths = list(input_signals.values())[0].wavelength
            N = _input_amplitude.shape[0]
            L = _input_amplitude.shape[1]
            M = len(input_port_names)
            
            outputs[output_port_name] = BlockModeOpticalSignal(amplitude=jnp.zeros((N, L, M), dtype=complex), wavelength=wavelengths)

            for mode_no, in_port_name in enumerate(input_port_names):
                input_amplitude = input_signals[in_port_name].amplitude[:, :, mode_no]
                wavelength = input_signals[in_port_name].wavelength
                output_amplitude = outputs[output_port_name].amplitude[:, :, mode_no] + input_amplitude
                outputs[output_port_name] = BlockModeOpticalSignal(amplitude=output_amplitude.reshape((N, L, M)), wavelength=wavelengths)

            return outputs


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
        
        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            **kwargs,
        ):
            self.output_port_names = output_port_names
            self.input_port_name = input_port_name
    
        # TODO: TEST THIS FOR MULTIPLE MODES
        def block_mode_response(self, inputs, simulation_parameters):
            input_amplitude = inputs[self.input_port_name].amplitude
            wl = inputs[self.input_port_name].wavelength
            
            outputs = {}
            for i, p in enumerate(self.output_port_names):
                output_amplitude = jnp.zeros((input_amplitude.shape[0], input_amplitude.shape[1], len(simulation_parameters.mode_identifiers)), dtype=complex)
                output_amplitude = output_amplitude.at[:, :, i].set(input_amplitude[:, :, i])
                outputs[p] = BlockModeOpticalSignal(amplitude=output_amplitude, wavelength=wl)
            
            return outputs
    
    return ModeDemultiplexer
