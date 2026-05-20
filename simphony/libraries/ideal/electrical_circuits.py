from simphony.component.component import SteadyStateComponent, SampleModeComponent, BlockModeComponent
from simphony.component.port import Port
from simphony.signal.steady_state import SteadyStateOpticalSignal, SteadyStateElectricalSignal
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from simphony.simulation.simulation import SimulationParameters

class VoltageFollower(
    SteadyStateComponent, 
    SampleModeComponent, 
    BlockModeComponent,
):
    """Pass an electrical signal from input port `e0` to output port `e1`.

    This helper is useful in directed Block mode netlists when an electrical
    signal needs an explicit through component. The Block mode response returns
    the input electrical signal object unchanged.
    """
    ports = [
        Port(
            name="e0",
            type="electrical",
            directionality="input",
        ),
        Port(
            name="e1",
            type="electrical",
            directionality="output",
        )
    ]

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
    ):
        pass

    def steady_state(
        self, 
        inputs: dict,
        simulation_parameters: SimulationParameters,
    ):
        outputs = {
            "e1": inputs["e0"]
        }
        return outputs

    def block_mode_response(self, input_signals: ArrayLike, simulation_parameters):
        return {
            "e1": input_signals["e0"]
        }
    
    def sample_mode_step(self, inputs: dict, state: jax.Array, simulation_state, simulation_parameters):
        # TODO: Complete this to use the signal defined in settings
        return inputs, state
    
    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])
