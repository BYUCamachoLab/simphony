# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import os

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from simphony.elements import Model
from simphony.tools import freq2wl, interpolate, wl2freq
from numpy import ndarray


class ebeam_bdc_te1550(Model):
    """A bidirectional coupler optimized for TE polarized light at 1550
    nanometers.

    The bidirectional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /user/libraries/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png
    """

    pins = ("n1", "n2", "n3", "n4")  #: The default pin names of the device
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sparams",
            "ebeam_bdc_te1550.npz",
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_dc_halfring_te1550(Model):
    pins = (
        "n1",
        "n2",
    )  #: The default pin names of the device
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sparams",
            "ebeam_dc_halfring_te1550.npz",
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_gc_te1550(Model):
    """A grating coupler optimized for TE polarized light at 1550 nanometers.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees
    [needs citation].

    .. image:: /user/libraries/images/ebeam_gc_te1550.png
        :alt: ebeam_bdc_te1550.png
    """

    pins = (
        "n1",
        "n2",
    )  #: The default pin names of the device
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sparams",
            "ebeam_gc_te1550.npz",
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_terminator_te1550(Model):
    """A terminator component that dissipates light into free space optimized
    for TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /user/libraries/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png
    """

    pins = ("n1",)  #: The default pin names of the device
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sparams",
            "ebeam_terminator_te1550.npz",
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_wg_integral_1550(Model):
    """Model for an waveguide optimized for TE polarized light at 1550
    nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /user/libraries/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    length : float
        Waveguide length in meters.
    lam0 : float, optional
        Central wavelength for calculation in meters (default 1.55 microns).
    ne : float, optional
        Effective index (default 2.44553).
    ng : float, optional
        Group velocity (default 4.19088).
    nd : float, optional
        Group dispersion (default 3.54275e-04).
    sigma_ne : float, optional
        Standard deviation of the effective index (default 0.05).
    sigma_ng : float, optional
        Standard deviation of the group velocity (default 0.05).
    sigma_nd : float, optional
        Standard deviation of the group dispersion (default 0.0001).

    Notes
    -----
    The ``sigma_`` values in the parameters are used for monte carlo simulations.
    """

    pins = (
        "n1",
        "n2",
    )  #: The default pin names of the device
    freq_range = (
        187370000000000.0,
        199862000000000.0,
    )  #: The valid frequency range for this model.

    def __init__(
        self,
        length: float,
        lam0: float=1.55e-06,
        ne: float=2.44553,
        ng: float=4.19088,
        nd: float=0.000354275,
        sigma_ne: float=0.05,
        sigma_ng: float=0.05,
        sigma_nd: float=0.0001,
    ) -> None:
        self.length = length
        self.lam0 = lam0
        self.ne = ne
        self.ng = ng
        self.nd = nd
        self.sigma_ne = sigma_ne
        self.sigma_ng = sigma_ng
        self.sigma_nd = sigma_nd
        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq: ndarray) -> ndarray:
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        start : float
            The starting frequency to obtain s-parameters for (in Hz).
        stop : float
            The ending frequency to obtain s-parameters for (in Hz).
        num : int
            The number of points to use between start_freq and stop_freq.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, ``frequency``,
            corresponding to the calculated s-parameter matrix, ``s``.
        """
        return self.cacl_s_params(
            freq, self.length, self.lam0, self.ne, self.ng, self.nd
        )

    def monte_carlo_s_parameters(self, freq):
        """Returns a monte carlo (randomized) set of s-parameters.

        In this implementation of the monte carlo routine, random values
        are generated for ne, ng, and nd for each run through of the
        monte carlo simulation. This means that all waveguide elements
        throughout a single circuit will have the same (random) ne, ng,
        and nd values. Hence, there is correlated randomness in the
        monte carlo parameters but they are consistent within a single
        circuit.
        """
        return self.cacl_s_params(
            freq, self.length, self.lam0, self.rand_ne, self.rand_ng, self.rand_nd
        )

    def regenerate_monte_carlo_parameters(self) -> None:
        self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
        self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
        self.rand_nd = np.random.normal(self.nd, self.sigma_nd)

    @staticmethod
    def cacl_s_params(frequency: ndarray, length: float, lam0: float, ne: float, ng: float, nd: float) -> ndarray:
        # Initialize array to hold s-params
        s = np.zeros((len(frequency), 2, 2), dtype=complex)

        # Loss calculation
        TE_loss = 700  # dB/m for width 500nm
        alpha = TE_loss / (20 * np.log10(np.exp(1)))

        w = np.asarray(frequency) * 2 * np.pi  # get angular frequency from frequency
        w0 = (2 * np.pi * SPEED_OF_LIGHT) / lam0  # center frequency (angular)

        # calculation of K
        K = (
            2 * np.pi * ne / lam0
            + (ng / SPEED_OF_LIGHT) * (w - w0)
            - (nd * lam0 ** 2 / (4 * np.pi * SPEED_OF_LIGHT)) * ((w - w0) ** 2)
        )

        for x in range(0, len(frequency)):  # build s-matrix from K and waveguide length
            s[x, 0, 1] = s[x, 1, 0] = np.exp(-alpha * length + (K[x] * length * 1j))

        return s


class ebeam_y_1550(Model):
    """The y-branch efficiently splits the input between the two outputs.

    .. image:: /user/libraries/images/ebeam_y_1550.png
        :alt: ebeam_bdc_te1550.png
    """

    pins = ("n1", "n2", "n3")  #: The default pin names of the device
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_y_1550.npz"
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_dc_te1550(Model):
    """A directional coupler optimized for TE polarized light at 1550
    nanometers.

    The directional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /user/libraries/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png
    """

    pins = ("n1", "n2", "n3", "n4")
    loaded = np.load(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sparams",
            "ebeam_dc_te1550.npz",
        )
    )
    s_params = (loaded["f"], loaded["s"])
    freq_range = (
        s_params[0][0],
        s_params[0][-1],
    )  #: The valid frequency range for this model.

    def s_parameters(self, freq: ndarray) -> ndarray:
        return interpolate(freq, self.s_params[0], self.s_params[1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bdc = ebeam_bdc_te1550()
    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    s = bdc.s_parameters(freq=f)
    plt.plot(wav, np.abs(s[:, 1] ** 2))

    plt.show()
