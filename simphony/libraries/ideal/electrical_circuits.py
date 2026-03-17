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

    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters):
        pass
    
    def sample_mode_step(self, inputs: dict, state: jax.Array, simulation_parameters):
        # TODO: Complete this to use the signal defined in settings
        return inputs, state
    
    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])