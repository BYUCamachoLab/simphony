import jax.numpy as jnp
from simphony.component.port import Port
from simphony.component.component import BlockModeComponent
from simphony.signal.block_mode import BlockModeOpticalSignal

class Terminator(BlockModeComponent):
    ports = [
        Port(name="o0", type="optical", directionality="bidirectional"),
    ]

    def __init__(self, simulation_parameters):
        pass

    def block_mode_response(self, inputs, simulation_parameters):
        T = simulation_parameters.num_time_steps
        wavelengths = simulation_parameters.optical_baseband_wavelengths
        L = wavelengths.shape[0]

        M = len(simulation_parameters.mode_identifiers)

        amp = jnp.zeros((T, L, M), dtype=jnp.complex64)

        return {
            "o0": BlockModeOpticalSignal(
                amplitude=amp,
                wavelength=wavelengths,
            )
        }