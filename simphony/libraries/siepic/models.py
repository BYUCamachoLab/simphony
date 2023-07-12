# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from pathlib import Path

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to py < 3.7 `importlib_resources`.
    import importlib_resources as pkg_resources

import numpy as np
import pandas as pd
import skrf
from skrf.frequency import Frequency

import simphony
from simphony.models import Model
from simphony.plugins.lumerical import load_sparams
from simphony.utils import wl2freq

SOURCE_DATA_PATH = "libraries/siepic/source_data"


def resolve_source_filepath(filename: str) -> Path:
    """Gets the absolute path to the source data files relative to ``source_data/``.

    Parameters
    ----------
    filename : str
        The name of the file to be found.

    Returns
    -------
    filepath : str
        The absolute path to the file.
    """
    filepath = Path(SOURCE_DATA_PATH) / filename
    try:  # python >= 3.9
        return pkg_resources.files(simphony) / filepath
    except AttributeError:  # fall back to method deprecated in 3.11.
        return pkg_resources.path(simphony, filepath)


class BidirectionalCouplerTE(Model):
    """SiEPIC EBeam PDK bidirectional coupler model.

    A bidirectional coupler optimized for TE polarized light at 1550nm.

    The bidirectional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        Waveguide thickness, in nanometers (default 220). Valid values
        are 210, 220, or 230 nanometers.
    width : float, optional
        Waveguide width, in nanometers (default 500). Valid values are
        480, 500, or 520 nanometers.

    Notes
    -----
    See also the PDK documentation:
    https://github.com/SiEPIC/SiEPIC_EBeam_PDK/blob/master/Documentation/SiEPIC_EBeam_PDK%20-%20Components%20with%20Models.docx
    """

    ocount = 4

    def __init__(self, pol="te", thickness=220.0, width=500):
        if pol not in ["te", "tm"]:
            raise ValueError("'pol' must be one of 'te' or 'tm'")
        self.pol = pol

        if thickness not in [210.0, 220.0, 230.0]:
            raise ValueError("'thickness' must be one of 210, 220, or 230")
        self.thickness = str(int(thickness))

        if width not in [480.0, 500.0, 520.0]:
            raise ValueError("'width' must be one of 480, 500, or 520")
        self.width = str(int(width))

        self.datafile = (
            f"bdc_TE_source/bdc_Thickness ={self.thickness} width={self.width}.sparam"
        )
        self.parsed_data = None

    # TODO: Is there a way to make caching work, or loading of data files,
    # without loading once per instance?
    def s_params(self, wl):
        if self.parsed_data is None:
            file = resolve_source_filepath(self.datafile)
            self.parsed_data = load_sparams(file)

        header, data = self.parsed_data["header"], self.parsed_data["data"]

        f, s = None, None
        for (p_out, p_in), sdf in data.groupby(["port_out", "port_in"]):
            freq = sdf["freq"].values
            if f is None:
                f = freq
            else:
                if not np.allclose(f, freq):
                    raise ValueError("Frequency mismatch between arrays in datafile.")

            if s is None:
                s = np.zeros((len(f), 4, 4), dtype=np.complex128)

            s[:, p_out - 1, p_in - 1] = sdf["mag"].values * np.exp(
                1j * sdf["phase"].values
            )

        ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
        interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
        return ntwk.interpolate(interp_freq).s


class DirectionalCoupler(Model):
    """A directional coupler optimized for TE polarized light at 1550nm.

    The directional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    gap : float, optional
        Coupling gap distance, in meters (default 200 nanometers).
    Lc : float, optional
        Length of coupler, in meters (default 10 microns).
    """

    ocount = 4

    def __init__(self):
        raise NotImplementedError


class GratingCoupler(Model):
    """SiEPIC EBeam PDK grating coupler optimized for TE polarizations at
    1550nm.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees
    [needs citation].

    .. image:: /reference/images/ebeam_gc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the grating coupler. Must be either 'te' (default) or
        'tm'.
    thickness : float
        Thickness of the grating coupler silicon in nm. Must be one of 210,
        220, or 230. Useful for simulating manufacturing variability.
    dwidth : float
        Change in width from nominal of the gratings. Representative of
        manufacturing variability. Must be one of -20, 0, or 20.

    Raises
    ------
    ValueError
        If `pol` is not 'te' or 'tm'.
    ValueError
        If `thickness` is not one of 210, 220, or 230.
    ValueError
        If `dwidth` is not one of -20, 0, or 20.

    Notes
    -----
    See also the PDK documentation:
    https://github.com/SiEPIC/SiEPIC_EBeam_PDK/blob/master/Documentation/SiEPIC_EBeam_PDK%20-%20Components%20with%20Models.docx
    """

    ocount = 2

    def __init__(self, pol="te", thickness=220.0, dwidth=0):
        if pol not in ["te", "tm"]:
            raise ValueError("'pol' must be either 'te' or 'tm'")
        self.pol = pol.upper()

        if thickness not in [210.0, 220.0, 230.0]:
            raise ValueError("'thickness' must be one of 210.0, 220.0, or 230")
        self.thickness = str(int(thickness))

        if dwidth not in [-20.0, 0.0, 20.0]:
            raise ValueError("'dwidth' must be one of -20, 0, or 20")
        self.dwidth = str(int(dwidth))

        self.datafile = f"gc_source/GC_{self.pol}1550_thickness={self.thickness} deltaw={self.dwidth}.txt"

    def s_params(self, wl):
        path = resolve_source_filepath(self.datafile)
        arr = np.loadtxt(path)

        f = arr[:, 0]
        s11 = arr[:, 1] * np.exp(1j * arr[:, 2])
        s12 = arr[:, 3] * np.exp(1j * arr[:, 4])
        s21 = arr[:, 5] * np.exp(1j * arr[:, 6])
        s22 = arr[:, 7] * np.exp(1j * arr[:, 8])
        s = np.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)

        ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
        interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
        return ntwk.interpolate(interp_freq).s


class HalfRing(Model):
    """A half-ring resonator optimized for TE polarized light at 1550nm.

    The halfring has 4 ports, labeled as pictured.

    .. image:: /reference/images/halfring.png
        :alt: halfring.png

    Parameters
    ----------
    gap : float, optional
        Coupling distance between ring and straight waveguide in meters
        (default 30 nanometers).
    radius : float, optional
        Ring radius in meters (default 10 microns).
    width : float, optional
        Waveguide width in meters (default 500 nanometers).
    thickness : float, optional
        Waveguide thickness in meters (default 220 nanometers).
    coupler_length : float, optional
        Length of the coupling edge, squares out ring; in meters (default 0).
    """

    ocount = 4

    def __init__(self):
        raise NotImplementedError


class Taper(Model):
    pass


class Terminator(Model):
    """A terminator component that dissipates light into free space optimized
    for TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /reference/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the grating coupler. Must be either 'te' (default) or
        'tm'.
    """

    ocount = 1

    def __init__(self):
        # ebeam_terminator_te1550/nanotaper_w1=500,w2=60,L=10_TE.sparam
        # ebeam_terminator_tm1550/nanotaper_w1=500,w2=60,L=10_TM.sparam
        raise NotImplementedError


class Waveguide(Model):
    """Model for an waveguide optimized for TE polarized light at 1550
    nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /reference/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    length : float
        Waveguide length in meters (default 0.0 meters).
    width : float, optional
        Waveguide width in meters (default 500 nanometers).
    height : float, optional
        Waveguide height in meters (default 220 nanometers).
    polarization : str, optional
        Polarization of light in the waveguide; one of 'TE' (default) or 'TM'.
    sigma_ne : float, optional
        Standard deviation of the effective index for monte carlo simulations
        (default 0.05).
    sigma_ng : float, optional
        Standard deviation of the group velocity for monte carlo simulations
        (default 0.05).
    sigma_nd : float, optional
        Standard deviation of the group dispersion for monte carlo simulations
        (default 0.0001).

    Notes
    -----
    The `sigma_` values in the parameters are used for monte carlo simulations.
    """

    ocount = 2

    def __init__(self):
        raise NotImplementedError


# TODO: Make sure the linked image displays in the docs
# TODO: Make sure the width parameter is accurately documented
class YBranch(Model):
    """SiEPIC EBeam PDK Y-branch model.

    A y-branch efficiently splits the input 50/50 between the two outputs.
    It can also be used as a combiner if used in the opposite direction,
    combining and interfering the light from two inputs into the one output.

    .. image:: /reference/images/ebeam_y_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the y-branch. Must be either 'te' (default) or 'tm'.
    thickness : float, optional
        Waveguide thickness, in nanometers (default 220). Valid values
        are 210, 220, or 230 nanometers. Useful for simulating manufacturing
        variability.
    width : float, optional
        Waveguide width, in nanometers (default 500 nanometers). Valid values
        are 480, 500, or 520 nanometers.

    Notes
    -----
    See also the PDK documentation:
    https://github.com/SiEPIC/SiEPIC_EBeam_PDK/blob/master/Documentation/SiEPIC_EBeam_PDK%20-%20Components%20with%20Models.docx
    """

    ocount = 3
    _POL_MAPPING = {"te": 1, "tm": 2}

    def __init__(self, pol="te", thickness=220.0, width=500):
        if pol not in ["te", "tm"]:
            raise ValueError("'pol' must be one of 'te' or 'tm'")
        self.pol = pol

        if thickness not in [210.0, 220.0, 230.0]:
            raise ValueError("'thickness' must be one of 210, 220, or 230")
        self.thickness = str(int(thickness))

        if width not in [480.0, 500.0, 520.0]:
            raise ValueError("'width' must be one of 480, 500, or 520")
        self.width = str(int(width))

        self.datafile = f"y_branch_source/Ybranch_Thickness ={self.thickness} width={self.width}.sparam"
        self.parsed_data = None

    # TODO: Is there a way to make caching work, or loading of data files,
    # without loading once per instance?
    def s_params(self, wl):
        if self.parsed_data is None:
            file = resolve_source_filepath(self.datafile)
            self.parsed_data = load_sparams(file)

        header, data = self.parsed_data["header"], self.parsed_data["data"]
        MODE_ID = self._POL_MAPPING[self.pol]

        f, s = None, None
        for (p_out, p_in), sdf in data[data.mode_out == MODE_ID].groupby(
            ["port_out", "port_in"]
        ):
            freq = sdf["freq"].values
            if f is None:
                f = freq
            else:
                if not np.allclose(f, freq):
                    raise ValueError("Frequency mismatch between arrays in datafile.")

            if s is None:
                s = np.zeros((len(f), 3, 3), dtype=np.complex128)

            s[:, p_out - 1, p_in - 1] = sdf["mag"].values * np.exp(
                1j * sdf["phase"].values
            )

        ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
        interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
        return ntwk.interpolate(interp_freq).s
