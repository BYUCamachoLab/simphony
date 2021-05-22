# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from typing import Optional, Tuple

import numpy as np

from simphony import Model
from simphony.pins import PinList
from simphony.tools import freq2wl, wl2freq


class Simulator(Model):
    pins = ("input", "output")

    def _prepare_simulation(self, freqs: np.array) -> Tuple[np.ndarray, PinList]:
        """Prepare the circuit to be simulated.

        This method gets the scattering parameters for the circuit and
        returns them with the list of corresponding pins.
        """
        if (
            not self.pins["input"]._isconnected()
            or not self.pins["output"]._isconnected()
        ):
            raise RuntimeError("Simulator must be connected before simulating.")

        subcircuit = self.circuit.to_subcircuit(rename_pins=True)
        s_params = subcircuit.s_parameters(freqs)

        return (s_params, subcircuit.pins)

    def simulate(
        self, *, freqs: Optional[np.array] = None, freq: float = 0, dB: bool = False
    ) -> Tuple[np.array, np.array]:
        """Simulates the circuit.

        Returns the list of frequencies with its corresponding scattering
        parameter.

        Parameters
        ----------
        freq :
            The single frequency to run the simulation for. Must be set if freqs
            is not.
        freqs :
            The list of frequencies to run simulations for. Must be set if freq
            is not.
        dB :
            Returns the scattering parameters in deciBels when True.
        """
        if freq:
            freqs = np.array(freq)

        s_params, pins = self._prepare_simulation(freqs)
        s_params = np.abs(s_params) ** 2

        if dB:
            s_params = np.log10(s_params)

        input = pins.index(self.pins["input"]._connection)
        output = pins.index(self.pins["output"]._connection)

        return (freqs, s_params[:, input, output])


class SweepSimulator(Simulator):
    """Wrapper simulator to make it easier to simulate over a range of
    frequencies."""

    def __init__(self, start: float = 1.5e-6, stop: float = 1.6e-6, num: int = 2000):
        """Initializes the SweepSimulator instance.

        The start and stop values can be given in either wavelength or
        frequency. The simulation will output results in the same mode.

        Parameters
        ----------
        start :
            The starting frequency/wavelength.
        stop :
            The stopping frequency/wavelength.
        num :
            The number of points between start and stop.
        """
        super().__init__()

        # automatically detect mode
        self.mode = "wl" if start < 1 else "freq"

        # if mode is wavelength, convert to frequencies
        if self.mode == "wl":
            start = wl2freq(start)
            stop = wl2freq(stop)

        self.freqs = np.linspace(start, stop, num)

    def simulate(self, **kwargs):
        """Runs the sweep simulation for the circuit."""
        freqs, s_params = super().simulate(**kwargs, freqs=self.freqs)

        if self.mode == "wl":
            return (freq2wl(freqs), s_params)

        return (freqs, s_params)
