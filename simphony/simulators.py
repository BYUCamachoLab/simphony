# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.simulators
===================

This module contains the simulator components. Simulators must be connected to
components before simulating. This can be done using the same connection methods
that exist on a component.
"""

from typing import List, Optional, Tuple

import numpy as np
import scipy

from simphony import Model
from simphony.libraries import siepic
from simphony.tools import freq2wl, wl2freq


class Simulator(Model):
    """Simulator model that can be used to instantiate a simulator.

    The first pin of the simulator should be attached to the input of
    the circuit that you want to simulate. The second pin should be
    attached to the output.
    """

    pins = ("to_input", "to_output")
    scache = {}

    def _generate(
        self,
        freqs: np.array,
        s_parameters_method: str = "s_parameters",
    ) -> np.ndarray:
        """Generates the scattering parameters for the circuit."""
        subcircuit = self.circuit.to_subcircuit(permanent=False)

        lower, upper = subcircuit.freq_range
        if lower > freqs[0] or upper < freqs[-1]:
            raise ValueError(
                f"Cannot simulate the range ({freqs[0], freqs[1]}) over the valid range ({lower}, {upper})"
            )

        return getattr(subcircuit, s_parameters_method)(freqs)

    def simulate(
        self,
        *,
        dB: bool = False,
        freq: float = 0,
        freqs: Optional[np.array] = None,
        s_parameters_method: str = "s_parameters",
    ) -> Tuple[np.array, np.array]:
        """Simulates the circuit.

        Returns the power ratio at each specified frequency.

        Parameters
        ----------
        dB :
            Returns the power ratios in deciBels when True.
        freq :
            The single frequency to run the simulation for. Must be set if freqs
            is not.
        freqs :
            The list of frequencies to run simulations for. Must be set if freq
            is not.
        s_parameters_method :
            The method name to call to get the scattering parameters.
        """
        if (
            not self.pins["to_input"]._isconnected()
            or not self.pins["to_output"]._isconnected()
        ):
            raise RuntimeError("Simulator must be connected before simulating.")

        # make sure we are working with an array of frequencies
        if freq:
            freqs = np.array(freq)

        # if the scattering parameters for the circuit are cached, use those
        try:
            if s_parameters_method == "monte_carlo_s_parameters" or "layout_aware_monte_carlo_s_parameters":
                raise RuntimeError("No caching for Monte Carlo simulations.")

            s_params = self.__class__.scache[self.circuit]
        except (KeyError, RuntimeError):
            s_params = self._generate(freqs, s_parameters_method)
            if s_parameters_method == "s_parameters":
                self.__class__.scache[self.circuit] = s_params

        # convert the scattering parameters to power ratios
        power_ratios = np.abs(s_params.copy()) ** 2
        if dB:
            power_ratios = np.log10(power_ratios)

        input = self.circuit.pins.index(self.pins["to_input"]._connection)
        output = self.circuit.pins.index(self.pins["to_output"]._connection)

        return (freqs, power_ratios[:, input, output])

    @classmethod
    def clear_scache(cls) -> None:
        """Clears the scattering parameters cache."""
        cls.scache = {}


class SweepSimulator(Simulator):
    """Wrapper simulator to make it easier to simulate over a range of
    frequencies."""

    def __init__(
        self, start: float = 1.5e-6, stop: float = 1.6e-6, num: int = 2000
    ) -> None:
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
            temp_start = start
            start = wl2freq(stop)
            stop = wl2freq(temp_start)

        if start > stop:
            raise ValueError(
                "Starting frequency cannot be greater than stopping frequency."
            )

        self.freqs = np.linspace(start, stop, num)

    def simulate(
        self, mode: Optional[str] = None, **kwargs
    ) -> Tuple[np.array, np.array]:
        """Runs the sweep simulation for the circuit.

        Parameters
        ----------
        dB :
            Returns the power ratios in deciBels when True.
        mode :
            Whether to return frequencies or wavelengths for the corresponding
            power ratios. Defaults to whatever values were passed in upon
            instantiation. Either 'freq' or 'wl'.
        """
        freqs, power_ratios = super().simulate(**kwargs, freqs=self.freqs)

        mode = mode if mode else self.mode
        if mode == "wl":
            return (freq2wl(freqs), power_ratios)

        return (freqs, power_ratios)


class MonteCarloSweepSimulator(SweepSimulator):
    """Wrapper simulator to make it easier to simulate over a range of
    frequencies while performing Monte Carlo experimentation."""

    def simulate(self, runs: int = 10, **kwargs) -> Tuple[np.array, np.array]:
        """Runs the Monte Carlo sweep simulation for the circuit.

        Parameters
        ----------
        dB :
            Returns the power ratios in deciBels when True.
        mode :
            Whether to return frequencies or wavelengths for the corresponding
            power ratios. Defaults to whatever values were passed in upon
            instantiation.
        runs :
            The number of Monte Carlo iterations to run (default 10).
        """
        results = []

        for i in range(runs):
            # use s_parameters for the first run, then monte_carlo_* for the rest
            s_parameters_method = "monte_carlo_s_parameters" if i else "s_parameters"
            results.append(
                super().simulate(**kwargs, s_parameters_method=s_parameters_method)
            )

            for component in self.circuit:
                component.regenerate_monte_carlo_parameters()

        return results


class LayoutAwareMonteCarloSweepSimulator(SweepSimulator):
    """Wrapper simulator to make it easier to simulate over a range of
    frequencies while performing Monte Carlo experimentation."""

    def simulate(self, x : List = [], y : List = [], sigmaw : float = 5, sigmat : float = 3, l : float = 4.5e-3, runs: int = 10, **kwargs) -> Tuple[np.array, np.array]:
        """Runs the Monte Carlo sweep simulation for the circuit.

        Parameters
        ----------
        dB :
            Returns the power ratios in deciBels when True.
        mode :
            Whether to return frequencies or wavelengths for the corresponding
            power ratios. Defaults to whatever values were passed in upon
            instantiation.
        runs :
            The number of Monte Carlo iterations to run (default 10).
        """
        results = []
        n = len(self.circuit._get_components())
        corr_matrix_w = np.zeros((n, n))
        corr_matrix_t = np.zeros((n, n))
        
        for i in range(n):
            for k in range(n):

                corr_val = np.exp(- ((x[k] - x[i]) ** 2 + (y[k] - y[i]) ** 2) / (0.5 * (l ** 2)))

                corr_matrix_w[i][k] = corr_matrix_w[k][i] = corr_val
                corr_matrix_t[i][k] = corr_matrix_t[k][i] = corr_val

        cov_matrix_w = np.zeros((n, n))
        cov_matrix_t = np.zeros((n, n))
        for i in range(n):
            for k in range(n):
                cov_matrix_w[i][k] = sigmaw * corr_matrix_w[i][k] * sigmaw
                cov_matrix_t[i][k] = sigmat * corr_matrix_t[i][k] * sigmat

        l_w = scipy.linalg.cholesky(cov_matrix_w, lower=True)
        l_t = scipy.linalg.cholesky(cov_matrix_t, lower=True)

        X = np.random.multivariate_normal(np.zeros(n), np.eye(n, n), runs).T

        corr_sample_matrix_w = np.dot(l_w, X)
        corr_sample_matrix_t = np.dot(l_t, X)

        components = self.circuit._get_components()

        for i in range(runs):
            # use s_parameters for the first run, then monte_carlo_* for the rest

            for idx, _ in enumerate(components):
                dw = corr_sample_matrix_w[idx][i]
                dt = corr_sample_matrix_t[idx][i]
                if isinstance(components[idx], siepic.Waveguide):
                    components[idx].__setattr__("layout_aware", True)
                    components[idx].__setattr__("width", components[idx].__getattribute__("width") + dw * 1e-9)
                    components[idx].__setattr__("height", components[idx].__getattribute__("height") + dt * 1e-9)
                    s_parameters_method = "layout_aware_monte_carlo_s_parameters" if i else "s_parameters"

            results.append(
                super().simulate(**kwargs, s_parameters_method=s_parameters_method)
            )

            for component in self.circuit:
                component.regenerate_monte_carlo_parameters()

        return results
