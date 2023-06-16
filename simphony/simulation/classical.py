"""
Module for classical simulation.
"""
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False

from .simulation import Simulation, SimulationResult
from .simdevices import Laser, Detector
from simphony.circuit import Circuit


@dataclass
class ClassicalResult(SimulationResult):
    """
    Classical simulation results
    """

    s_params: jnp.ndarray
    input_source: jnp.ndarray
    output: jnp.ndarray
    detectors: dict
    wl: jnp.ndarray


class ClassicalSim(Simulation):
    """
    Classical simulation
    """

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        """
        Initialize the classical simulation.

        Parameters
        ----------
        ckt : Circuit
            The circuit to simulate.
        wl : jnp.ndarray
            The wavelengths to simulate.
        """
        super().__init__(ckt, wl)

        # find the index of all laser simdevices
        self.laser_dict = {}
        for laser in ckt.sim_devices:
            if isinstance(laser, Laser):
                laser_idx = ckt._oports.index(laser.ports[0])
                self.laser_dict[laser] = laser_idx

        # find the index of all detector simdevices
        self.detector_dict = {}
        for detector in ckt.sim_devices:
            if isinstance(detector, Detector):
                detector_idx = ckt._oports.index(detector.ports[0])
                self.detector_dict[detector] = detector_idx

    def run(self) -> ClassicalResult:
        """
        Run the classical simulation.
        """

        # Get the S-matrix for the circuit
        S = self.ckt.s_params(self.wl)

        # Get input array from all lasers
        input_source = jnp.zeros(
            (len(self.wl), len(self.ckt._oports)), dtype=jnp.complex128
        )
        for laser, laser_idx in self.laser_dict.items():
            if laser.mod_function is None:
                input_source[:, laser_idx] = (
                    jnp.ones_like(self.wl)
                    * jnp.sqrt(laser.power)
                    * jnp.exp(1j * laser.phase)
                )
            else:
                input_source[:, laser_idx] = laser.mod_function(self.wl) * jnp.sqrt(
                    laser.power
                )

        # Calculate the output from all detectors
        output = (S @ input_source[:, :, None])[:, :, 0]
        # TODO: This could be optimized by only using the rows corresponding to
        # the detectors

        result = ClassicalResult(
            s_params=S,
            input_source=input_source,
            output=output,
            detectors=self.detector_dict,
            wl=self.wl,
        )

        return result
