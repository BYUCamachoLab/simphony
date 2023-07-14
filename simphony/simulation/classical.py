"""Module for classical simulation."""
from dataclasses import dataclass
from typing import Callable

from charset_normalizer import detect

from simphony.models import OPort

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    from simphony.utils import jax

    JAX_AVAILABLE = False

from simphony.circuit import Circuit

from .simdevices import Detector, Laser
from .simulation import Simulation, SimulationResult


@dataclass
class ClassicalResult(SimulationResult):
    """Classical simulation results.

    Attributes
    ----------
    wl : jnp.ndarray
        The wavelengths at which the simulation was run.
    s_params : jnp.ndarray
        The S-parameters of the circuit.
    input_source : jnp.ndarray
        The input source at each wavelength.
    output : jnp.ndarray
        The output at each wavelength.
    detectors : list[Detector]
        The detectors and their measurements from the simulation. They are
        indexed in the same order as both ``s_params`` and ``output``.
    """

    wl: jnp.ndarray
    s_params: jnp.ndarray
    input_source: jnp.ndarray
    output: jnp.ndarray
    detectors: list[Detector]


class ClassicalSim(Simulation):
    """Classical simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
        """Initialize the classical simulation.

        Parameters
        ----------
        ckt : Circuit
            The circuit to simulate.
        wl : jnp.ndarray
            The array of wavelengths to simulate (in microns).
        """
        super().__init__(ckt, wl)
        self.lasers: dict[Laser, list[OPort]] = {}
        self.detectors: dict[Detector, list[OPort]] = {}

    def add_laser(
        self,
        ports: OPort | list[OPort],
        power: float = 1.0,
        phase: float = 0.0,
        mod_function: Callable = None,
    ) -> Laser:
        """Add an ideal laser source.

        If multiple ports are specified, the same laser will be connected
        to all of them.

        Parameters
        ----------
        ports : OPort | list[OPort]
            The ports to which the laser is connected.
        power : float, optional
            The power of the laser (in mW), by default 1.0
        phase : float, optional
            The phase of the laser (in radians), by default 0.0
        mod_function : Callable, optional
            The modulation function, by default None (not yet implemented).
        """
        ports = list(ports)
        laser = Laser(ports, power, phase, mod_function)
        self.lasers[laser] = ports
        return laser

    def add_detector(
        self, ports: OPort | list[OPort], responsivity: float = 1.0
    ) -> Detector | list[Detector]:
        """Add an ideal photodetector.

        If multiple ports are specified, multiple detectors will be created
        and returned.

        Parameters
        ----------
        ports : OPort | list[OPort]
            The ports to which the detector is connected.
        responsivity : float, optional
            The responsivity of the detector (in A/W), by default 1.0

        Returns
        -------
        Detector | list[Detector]
            The created detector(s).
        """
        ports = list(ports)
        detectors = []
        for port in ports:
            detector = Detector(port, responsivity)
            self.detectors[detector] = list(port)
            detectors.append(detector)
        return detectors

    def run(self) -> ClassicalResult:
        """Run the classical simulation.

        Returns
        -------
        ClassicalResult
            The simulation results.
        """

        # Get the S-matrix for the circuit
        if JAX_AVAILABLE:
            s = self.ckt._s(tuple(self.wl.tolist()))
        else:
            s = self.ckt._s(tuple(jnp.asarray(self.wl).reshape(-1)))

        # Create input vector from all lasers
        src_v = jnp.zeros((len(self.wl), len(self.ckt._oports)), dtype=jnp.complex128)
        for laser, ports in self.lasers.items():
            idx = [self.ckt._oports.index(port) for port in ports]
            if laser.mod_function is None:
                if JAX_AVAILABLE:
                    src_v = src_v.at[:, idx].set(
                        jnp.sqrt(laser.power) * jnp.exp(1j * laser.phase)
                    )
                else:
                    src_v[:, idx] = jnp.sqrt(laser.power) * jnp.exp(1j * laser.phase)
            else:
                raise NotImplementedError
                # if JAX_AVAILABLE:
                #     src_v = src_v.at[:,idx].set(laser.mod_function(self.wl) * jnp.sqrt(laser.power))
                # else:
                #     src_v[:, idx] = laser.mod_function(self.wl) * jnp.sqrt(laser.power)

        # Calculate the output from all detectors
        port_det_mapping: dict[OPort, Detector] = {
            port: detector
            for detector in self.detectors
            for port in self.detectors[detector]
        }
        # Build parallel arrays of indices and ports
        indices = []
        portarr = []
        for i, port in enumerate(self.ckt._oports):
            if port in port_det_mapping:
                indices.append(i)
                portarr.append(port)

        # Only calculate the output for ports with detectors
        output = (s[:, indices, :] @ src_v[:, :, None])[:, :, 0]

        # Return a list of detectors with their measurements, indexed in the
        # same order as "output".
        detectors = []
        for i, port in enumerate(portarr):
            detector = port_det_mapping[port]
            detector.set_result(
                wl=self.wl,
                power=(jnp.square(jnp.abs(output[:, i])) * detector.responsivity),
            )
            detectors.append(detector)

        result = ClassicalResult(
            wl=self.wl,
            s_params=s,
            input_source=src_v,
            output=output,
            detectors=detectors,
        )

        return result


# class MonteCarloSim(Simulation):
#     """Monte Carlo simulation."""

#     def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(ckt, wl)


# class LayoutAwareSim(Simulation):
#     """Layout-aware simulation."""

#     def __init__(self, cir: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(cir, wl)


# class SamplingSim(Simulation):
#     """Sampling simulation."""

#     def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(ckt, wl)


# class TimeDomainSim(Simulation):
#     """Time-domain simulation."""

#     def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(ckt, wl)


# class QuantumSim(Simulation):
#     """Quantum simulation."""

#     def __init__(self, ckt: Circuit, wl: jnp.ndarray) -> None:
#         super().__init__(ckt, wl)
