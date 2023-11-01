"""Module for simulation devices."""


import warnings
from typing import List, Union

import matplotlib.pyplot as plt
from jax.typing import ArrayLike


class SimDevice:
    """Base class for all source or measure devices."""

    def __init__(self, ports: list) -> None:
        self.ports = ports


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
        self.wl = wl
        self.power = power

    def plot(self, ax=None, **kwargs):
        """Plot the detector response."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.wl, self.power, **kwargs)
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Power (mW)")
        return ax
