"""Module for classical simulation."""
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model

from simphony.simulation import SimDevice, Simulation, SimulationResult


class Laser(SimDevice):
    """Ideal laser source.

    Parameters
    ----------
    ports : str or list of str
        The ports to which the laser is connected.
    power : float, optional
        The power of the laser (in mW), by default 1.0
    phase : float, optional
        The phase of the laser (in radians), by default 0.0
    mod_function : Callable, optional
        The modulation function, by default None.
    """

    def __init__(
        self,
        ports: Union[str, List[str]],
        power: float = 1.0,
        phase: float = 0.0,
        mod_function=None,
    ) -> None:
        super().__init__(list(ports))
        self.power = power
        self.phase = phase
        # TODO: Implement mod_function
        self.mod_function = mod_function


class Detector(SimDevice):
    """Ideal photodetector.

    Attributes
    ----------
    wl : jnp.ndarray
        The wavelengths at which the detector was simulated.
    power : jnp.ndarray
        The power at each wavelength.

    Parameters
    ----------
    port : str
        The port to which the detector is connected.
    responsivity : float, optional
        The responsivity of the detector (in A/W), by default 1.0
    """

    def __init__(self, port: str, responsivity: float = 1.0) -> None:
        super().__init__(list(port))
        if responsivity != 1.0:
            warnings.warn("Responsivity is not yet implemented, so it is ignored.")
        self.responsivity = responsivity

    def set_result(self, wl: ArrayLike, power: ArrayLike) -> None:
        """Set the result of the detector.

        Parameters
        ----------
        wl : ArrayLike
            The wavelengths at which the detector was simulated.
        power : ArrayLike
            The power at each wavelength.
        """
        self.wl = wl
        self.power = power

    def plot(self, ax=None, **kwargs):
        """Plot the detector response.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot, by default None (create new axes).
        **kwargs
            Any other keyword arguments to pass to matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.wl, self.power, **kwargs)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Power (mW)")
        return ax


@dataclass
class ClassicalResult(SimulationResult):
    """Classical simulation results.

    Attributes
    ----------
    wl : jnp.ndarray
        The wavelengths at which the simulation was run.
    sdict : sax.SDict
        The S-parameters of the circuit.
    detectors : list[Detector]
        The detectors and their measurements from the simulation. They are
        indexed in the same order as both ``s_params`` and ``output``.
    """

    # TODO: Add a function to convert the sdict to a matrix and easily get
    # port ordering.

    wl: ArrayLike
    sdict: sax.SDict
    detectors: list[Detector]


class ClassicalSim(Simulation):
    """Classical simulation."""

    def __init__(self, ckt: Model, **kwargs) -> None:
        """Initialize the classical simulation.

        Parameters
        ----------
        ckt : sax.saxtypes.Model
            The circuit to simulate.
        wl : ArrayLike
            The array of wavelengths to simulate (in microns).
        **params
            Any other parameters to pass to the circuit.

        Examples
        --------
        >>> sim = ClassicalSim(ckt=mzi, wl=wl, top={"length": 150.0}, bottom={"length": 50.0})
        """
        # TODO: Add shot noise option to classical simulation
        ckt = partial(ckt, **kwargs)
        if "wl" not in kwargs:
            raise ValueError("Must specify 'wl' (wavelengths to simulate).")
        super().__init__(ckt, kwargs["wl"])
        self.lasers: dict[str, Laser] = {}
        self.detectors: dict[str, Detector] = {}

    def add_laser(
        self,
        ports: Union[str, List[str]],
        power: float = 1.0,
        phase: float = 0.0,
        mod_function: Callable = None,
    ) -> Laser:
        """Add an ideal laser source.

        If multiple ports are specified, the same laser will be connected
        to all of them.

        Parameters
        ----------
        ports : OPort or list of OPort
            The ports to which the laser is connected.
        power : float, optional
            The power of the laser (in mW), by default 1.0
        phase : float, optional
            The phase of the laser (in radians), by default 0.0
        mod_function : Callable, optional
            The modulation function, by default None (not yet implemented).

        Returns
        -------
        Laser
            The created laser.

        Examples
        --------
        >>> laser = sim.add_laser(ports=["in"], power=1.0)
        """
        ports = [ports] if not isinstance(ports, list) else ports
        laser = Laser(ports, power, phase, mod_function)
        for port in ports:
            self.lasers[port] = laser
        return laser

    def add_detector(
        self, ports: Union[str, List[str]], responsivity: float = 1.0
    ) -> List[Detector]:
        """Add an ideal photodetector.

        If multiple ports are specified, multiple detectors will be created
        and returned.

        Parameters
        ----------
        ports : OPort or list of OPort
            The ports to which the detector is connected.
        responsivity : float, optional
            The responsivity of the detector (in A/W), by default 1.0

        Returns
        -------
        list of Detector
            A list of the created detector(s) (potentially a list of length 1).

        Examples
        --------
        >>> detector = sim.add_detector(ports=["out"], responsivity=0.8)
        """
        ports = [ports] if not isinstance(ports, list) else ports
        detectors = []
        for port in ports:
            detector = Detector(port, responsivity)
            self.detectors[port] = detector
            detectors.append(detector)
        return detectors

    def run(self) -> ClassicalResult:
        """Run the classical simulation.

        Returns
        -------
        ClassicalResult
            The simulation results.
        """
        S = self.ckt()

        sdict = {}
        for output_port in self.detectors:
            responses = []
            for input_port in self.lasers:
                signal = jnp.sqrt(self.lasers[input_port].power) * jnp.exp(
                    1j * self.lasers[input_port].phase
                )
                responses.append(S[output_port, input_port] * signal)
            sdict[output_port] = jnp.sum(jnp.asarray(responses), axis=0)

        # # Create input vector from all lasers
        # src_v = jnp.zeros((len(self.wl), len(ports)), dtype=jnp.complex64)
        # for laser, ports in self.lasers.items():
        #     idx = [self.ckt._oports.index(port) for port in ports]
        #     if laser.mod_function is None:
        #         src_v = src_v.at[:, idx].set(jnp.sqrt(laser.power) * jnp.exp(1j * laser.phase))
        #     else:
        #         raise NotImplementedError
        #         # src_v = src_v.at[:,idx].set(laser.mod_function(self.wl) * jnp.sqrt(laser.power))

        for port, detector in self.detectors.items():
            power = (jnp.abs(sdict[port]) ** 2) * detector.responsivity
            detector.set_result(wl=self.wl, power=power)

        result = ClassicalResult(
            wl=self.wl,
            sdict=sdict,
            detectors=self.detectors,
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
