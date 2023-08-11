# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""This package contains parameterized models of PIC components from the SiEPIC
Electron Beam Lithography Process Development Kit (PDK) from the University of
British Columbia (UBC), which is licensed under the terms of the MIT License.

See their repository for more details (
https://github.com/SiEPIC/SiEPIC_EBeam_PDK).
"""

import importlib.resources
import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import skrf
from scipy.constants import c as SPEED_OF_LIGHT
from skrf.frequency import Frequency
from tabulate import tabulate

import simphony.libraries
from simphony.models import Model
from simphony.plugins.lumerical import load_sparams
from simphony.utils import wl2freq

SOURCE_DATA_PATH = "siepic/source_data"


def _resolve_source_filepath(filename: str) -> Path:
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
        return importlib.resources.files(simphony.libraries) / filepath
    except AttributeError:  # fall back to method deprecated in 3.11.
        ctx = importlib.resources.path(simphony, "libraries")
        with ctx as path:
            return path / filepath


class BidirectionalCouplerTE(Model):
    """SiEPIC EBeam PDK bidirectional coupler model.

    A bidirectional coupler optimized for TE polarized light at 1550nm.

    The bidirectional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /_static/images/ebeam_bdc_te1550.png
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

    def s_params(self, wl):
        file = _resolve_source_filepath(self.datafile)
        header, data = load_sparams(file)

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

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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

    .. image:: /_static/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    gap : float, optional
        Coupling gap distance, in nanometers (default 200).
    coupling_length : float, optional
        Length of coupler, in microns (default 10).

    Notes
    -----
    Sorted matrix of valid parameter sets for directional couplers:

    =====  =================
    gap    coupling_length
    =====  =================
    200                  0
    200                  5
    200                 10
    200                 15
    200                 20
    200                 25
    200                 30
    200                 35
    200                 40
    200                 45
    =====  =================
    """

    ocount = 4

    def __init__(self, gap=200, coupling_length=0):
        df = self._generate_parameter_sets()
        if not ((df["gap"] == gap) & (df["coupling_length"] == coupling_length)).any():
            raise ValueError(
                "Invalid parameter set, see the documentation for valid parameter sets"
            )

        self.gap = gap
        self.coupling_length = coupling_length

        self._datafile = f"ebeam_dc_te1550/dc_gap={gap}nm_Lc={coupling_length}um.sparam"

    @classmethod
    def _generate_parameter_sets(cls) -> pd.DataFrame:
        """Generate a dataframe of all valid parameter sets by parsing the
        filenames of the data files in the source directory."""
        pattern = r"dc_gap=(?P<gap>\d+)nm_Lc=(?P<coupling_length>\d+)um.sparam"
        path = _resolve_source_filepath("ebeam_dc_te1550")

        params = []
        for file in [str(p.name) for p in path.glob("*.sparam")]:
            m = re.match(pattern, file)
            if m:
                params.append(m.groupdict())

        df = pd.DataFrame(params)
        df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
        df["coupling_length"] = pd.to_numeric(df["coupling_length"], errors="coerce")

        # Find out which columns have the most unique items
        unique_counts = df.nunique()
        # Sort dataframe by fewest unique items in column to most unique items
        sorted_values = unique_counts.sort_values().index.tolist()
        df = df.sort_values(by=sorted_values, ignore_index=True)
        # Rearrange columns of dataframe to go from fewest unique items to most
        sorted_df = df[sorted_values]

        return sorted_df

    @classmethod
    def _generate_parameter_table_rst(cls) -> str:
        """Generate a rst table of valid parameter sets for the docstring."""
        df = cls._generate_parameter_sets()
        df.reset_index(drop=True, inplace=True)
        return tabulate(df.values, df.columns, tablefmt="rst")

    def s_params(self, wl):
        file = _resolve_source_filepath(self._datafile)
        header, data = load_sparams(file)

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

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


class GratingCoupler(Model):
    """SiEPIC EBeam PDK grating coupler optimized for TE polarizations at
    1550nm.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees
    [needs citation].

    .. image:: /_static/images/ebeam_gc_te1550.png
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

        self._datafile = f"gc_source/GC_{self.pol}1550_thickness={self.thickness} deltaw={self.dwidth}.txt"

    def s_params(self, wl):
        path = _resolve_source_filepath(self._datafile)
        arr = np.loadtxt(path)

        f = arr[:, 0]
        s11 = arr[:, 1] * np.exp(1j * arr[:, 2])
        s12 = arr[:, 3] * np.exp(1j * arr[:, 4])
        s21 = arr[:, 5] * np.exp(1j * arr[:, 6])
        s22 = arr[:, 7] * np.exp(1j * arr[:, 8])
        s = np.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


class HalfRing(Model):
    """A half-ring resonator optimized for TE polarized light at 1550nm.

    The halfring has 4 ports, labeled as pictured.

    .. image:: /_static/images/halfring.png
        :alt: halfring.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the halfring. Must be either 'te' (default) or 'tm'.
    gap : float, optional
        Coupling distance between ring and straight waveguide in nanometers
        (default 50).
    radius : float, optional
        Ring radius in microns (default 5).
    width : float, optional
        Waveguide width in nanometers (default 500).
    thickness : float, optional
        Waveguide thickness in nanometers (default 220).
    coupling_length : float, optional
        Length of the straight segment of the directional coupling edge, turns
        ring into a racetrack resonator, in microns (default 0).

    Notes
    -----
    Sorted matrix of valid parameter sets for half rings:

    =====  =================  =======  ===========  ========  =====
    pol      coupling_length    width    thickness    radius    gap
    =====  =================  =======  ===========  ========  =====
    te                     0      480          210         3     70
    te                     0      480          210         3     80
    te                     0      480          210         3    100
    te                     0      480          210         3    120
    te                     0      480          210         5     70
    te                     0      480          210         5     80
    te                     0      480          210         5    120
    te                     0      480          210        10    120
    te                     0      480          210        10    170
    te                     0      480          230         3     70
    te                     0      480          230         3     80
    te                     0      480          230         3    100
    te                     0      480          230         3    120
    te                     0      480          230         5     70
    te                     0      480          230         5     80
    te                     0      480          230         5    120
    te                     0      480          230        10    120
    te                     0      480          230        10    170
    te                     0      500          220         3     50
    te                     0      500          220         3     60
    te                     0      500          220         3     80
    te                     0      500          220         3    100
    te                     0      500          220         5     50
    te                     0      500          220         5     60
    te                     0      500          220         5    100
    te                     0      500          220        10    100
    te                     0      500          220        10    150
    te                     0      500          220        18    200
    te                     0      520          210         3     30
    te                     0      520          210         3     40
    te                     0      520          210         3     60
    te                     0      520          210         3     80
    te                     0      520          210         5     30
    te                     0      520          210         5     40
    te                     0      520          210         5     80
    te                     0      520          210        10     80
    te                     0      520          210        10    130
    te                     0      520          230         3     30
    te                     0      520          230         3     40
    te                     0      520          230         3     60
    te                     0      520          230         3     80
    te                     0      520          230         5     30
    te                     0      520          230         5     40
    te                     0      520          230         5     80
    te                     0      520          230        10     80
    te                     0      520          230        10    130
    te                     4      500          220        10    200
    tm                     0      480          210         5    320
    tm                     0      480          230         5    320
    tm                     0      500          220         5    300
    tm                     0      520          210         5    280
    tm                     0      520          230         5    280
    =====  =================  =======  ===========  ========  =====
    """

    ocount = 4

    def __init__(
        self, pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
    ):
        if pol not in ["te", "tm"]:
            raise ValueError("'pol' must be one of 'te' or 'tm'")

        df = self._generate_parameter_sets()
        if not (
            (df["pol"] == pol)
            & (df["gap"] == gap)
            & (df["radius"] == radius)
            & (df["width"] == width)
            & (df["thickness"] == thickness)
            & (df["coupling_length"] == coupling_length)
        ).any():
            raise ValueError(
                "Invalid parameter set, see the documentation for valid parameter sets"
            )
        # (df == a).all(1).any()

        self.pol = pol
        self.gap = gap
        self.radius = radius
        self.width = width
        self.thickness = thickness
        self.coupling_length = coupling_length

        # ebeam_dc_halfring_straight/te_ebeam_dc_halfring_straight_gap=30nm_radius=3um_width=520nm_thickness=210nm_CoupleLength=0um.dat
        self._datafile = f"ebeam_dc_halfring_straight/{pol}_ebeam_dc_halfring_straight_gap={int(gap)}nm_radius={int(radius)}um_width={int(width)}nm_thickness={int(thickness)}nm_CoupleLength={int(coupling_length)}um.dat"

    @classmethod
    def _generate_parameter_sets(cls) -> pd.DataFrame:
        """Generate a dataframe of all valid parameter sets by parsing the
        filenames of the data files in the source directory."""
        pattern = r"(?P<pol>[a-z]+)_ebeam_dc_halfring_straight_gap=(?P<gap>\d+)nm_radius=(?P<radius>\d+)um_width=(?P<width>\d+)nm_thickness=(?P<thickness>\d+)nm_CoupleLength=(?P<coupling_length>\d+)um.dat"
        path = _resolve_source_filepath("ebeam_dc_halfring_straight")

        params = []
        for file in [str(p.name) for p in path.glob("*.dat")]:
            m = re.match(pattern, file)
            if m:
                params.append(m.groupdict())

        df = pd.DataFrame(params)
        df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
        df["width"] = pd.to_numeric(df["width"], errors="coerce")
        df["thickness"] = pd.to_numeric(df["thickness"], errors="coerce")
        df["coupling_length"] = pd.to_numeric(df["coupling_length"], errors="coerce")

        # Find out which columns have the most unique items
        unique_counts = df.nunique()
        # Sort dataframe by fewest unique items in column to most unique items
        sorted_values = unique_counts.sort_values().index.tolist()
        df = df.sort_values(by=sorted_values, ignore_index=True)
        # Rearrange columns of dataframe to go from fewest unique items to most
        sorted_df = df[sorted_values]

        return sorted_df

    @classmethod
    def _generate_parameter_table_rst(cls) -> str:
        """Generate a rst table of valid parameter sets for the docstring."""
        df = cls._generate_parameter_sets()
        df.reset_index(drop=True, inplace=True)
        return tabulate(df.values, df.columns, tablefmt="rst")

    def s_params(self, wl):
        file = _resolve_source_filepath(self._datafile)
        header, data = load_sparams(file)

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

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


class Taper(Model):
    """A taper component that adiabatically transitions between two waveguide
    widths.

    This taper is simulated for TE operation at 1550 nanometers.

    .. image:: /_static/images/ebeam_taper_te1550.png
        :alt: ebeam_taper_te1550.png

    Parameters
    ----------
    w1 : float, optional
        Width of the input waveguide in microns (default 0.5).
    w2 : float, optional
        Width of the output waveguide in microns (default 1).
    length : float, optional
        Length of the taper in microns (default 10).

    Notes
    -----
    Sorted matrix of valid parameter sets for adiabatic tapers:

    ====  ====  ========
    w1    w2    length
    ====  ====  ========
    0.4     1         1
    0.4     1         2
    0.4     1         3
    0.4     1         4
    0.4     1         5
    0.4     1         6
    0.4     1         7
    0.4     1         8
    0.4     1         9
    0.4     1        10
    0.4     1        11
    0.4     1        12
    0.4     1        13
    0.4     1        14
    0.4     1        15
    0.4     1        16
    0.4     1        17
    0.4     1        18
    0.4     1        19
    0.4     1        20
    0.4     2         1
    0.4     2         2
    0.4     2         3
    0.4     2         4
    0.4     2         5
    0.4     2         6
    0.4     2         7
    0.4     2         8
    0.4     2         9
    0.4     2        10
    0.4     2        11
    0.4     2        12
    0.4     2        13
    0.4     2        14
    0.4     2        15
    0.4     2        16
    0.4     2        17
    0.4     2        18
    0.4     2        19
    0.4     2        20
    0.4     3         1
    0.4     3         2
    0.4     3         3
    0.4     3         4
    0.4     3         5
    0.4     3         6
    0.4     3         7
    0.4     3         8
    0.4     3         9
    0.4     3        10
    0.4     3        11
    0.4     3        12
    0.4     3        13
    0.4     3        14
    0.4     3        15
    0.4     3        16
    0.4     3        17
    0.4     3        18
    0.4     3        19
    0.4     3        20
    0.5     1         1
    0.5     1         2
    0.5     1         3
    0.5     1         4
    0.5     1         5
    0.5     1         6
    0.5     1         7
    0.5     1         8
    0.5     1         9
    0.5     1        10
    0.5     1        11
    0.5     1        12
    0.5     1        13
    0.5     1        14
    0.5     1        15
    0.5     1        16
    0.5     1        17
    0.5     1        18
    0.5     1        19
    0.5     1        20
    0.5     2         1
    0.5     2         2
    0.5     2         3
    0.5     2         4
    0.5     2         5
    0.5     2         6
    0.5     2         7
    0.5     2         8
    0.5     2         9
    0.5     2        10
    0.5     2        11
    0.5     2        12
    0.5     2        13
    0.5     2        14
    0.5     2        15
    0.5     2        16
    0.5     2        17
    0.5     2        18
    0.5     2        19
    0.5     2        20
    0.5     3         1
    0.5     3         2
    0.5     3         3
    0.5     3         4
    0.5     3         5
    0.5     3         6
    0.5     3         7
    0.5     3         8
    0.5     3         9
    0.5     3        10
    0.5     3        11
    0.5     3        12
    0.5     3        13
    0.5     3        14
    0.5     3        15
    0.5     3        16
    0.5     3        17
    0.5     3        18
    0.5     3        19
    0.5     3        20
    0.6     1         1
    0.6     1         2
    0.6     1         3
    0.6     1         4
    0.6     1         5
    0.6     1         6
    0.6     1         7
    0.6     1         8
    0.6     1         9
    0.6     1        10
    0.6     1        11
    0.6     1        12
    0.6     1        13
    0.6     1        14
    0.6     1        15
    0.6     1        16
    0.6     1        17
    0.6     1        18
    0.6     1        19
    0.6     1        20
    0.6     2         1
    0.6     2         2
    0.6     2         3
    0.6     2         4
    0.6     2         5
    0.6     2         6
    0.6     2         7
    0.6     2         8
    0.6     2         9
    0.6     2        10
    0.6     2        11
    0.6     2        12
    0.6     2        13
    0.6     2        14
    0.6     2        15
    0.6     2        16
    0.6     2        17
    0.6     2        18
    0.6     2        19
    0.6     2        20
    0.6     3         1
    0.6     3         2
    0.6     3         3
    0.6     3         4
    0.6     3         5
    0.6     3         6
    0.6     3         7
    0.6     3         8
    0.6     3         9
    0.6     3        10
    0.6     3        11
    0.6     3        12
    0.6     3        13
    0.6     3        14
    0.6     3        15
    0.6     3        16
    0.6     3        17
    0.6     3        18
    0.6     3        19
    0.6     3        20
    ====  ====  ========
    """

    ocount = 2

    def __init__(self, w1: float = 0.5, w2: float = 1.0, length: float = 10.0):
        df = self._generate_parameter_sets()
        if not ((df["w1"] == w1) & (df["w2"] == w2) & (df["length"] == length)).any():
            raise ValueError(
                "Invalid parameter set, see the documentation for valid parameter sets"
            )

        self.w1 = w1
        self.w2 = w2
        self.length = length

        self._datafile = f"ebeam_taper_te1550/w1={w1:.1f}um_w2={int(w2)}um_length={int(length)}um.dat"

    @classmethod
    def _generate_parameter_sets(cls) -> pd.DataFrame:
        """Generate a dataframe of all valid parameter sets by parsing the
        filenames of the data files in the source directory."""
        pattern = r"w1=(?P<w1>\d+\.\d+)um_w2=(?P<w2>\d+)um_length=(?P<length>\d+)um.dat"
        path = _resolve_source_filepath("ebeam_taper_te1550")

        params = []
        for file in [str(p.name) for p in path.glob("*.dat")]:
            m = re.match(pattern, file)
            if m:
                params.append(m.groupdict())

        df = pd.DataFrame(params)
        df["w1"] = pd.to_numeric(df["w1"], errors="coerce")
        df["w2"] = pd.to_numeric(df["w2"], errors="coerce")
        df["length"] = pd.to_numeric(df["length"], errors="coerce")

        # Find out which columns have the most unique items
        unique_counts = df.nunique()
        # Sort dataframe by fewest unique items in column to most unique items
        sorted_values = unique_counts.sort_values().index.tolist()
        df = df.sort_values(by=sorted_values, ignore_index=True)
        # Rearrange columns of dataframe to go from fewest unique items to most
        sorted_df = df[sorted_values]

        return sorted_df

    @classmethod
    def _generate_parameter_table_rst(cls) -> str:
        """Generate a rst table of valid parameter sets for the docstring."""
        df = cls._generate_parameter_sets()
        df.reset_index(drop=True, inplace=True)
        return tabulate(df.values, df.columns, tablefmt="rst")

    def s_params(self, wl):
        path = _resolve_source_filepath(self._datafile)
        arr = np.loadtxt(path)

        f = arr[:, 0]
        s11 = arr[:, 1] * np.exp(1j * arr[:, 2])
        s12 = arr[:, 3] * np.exp(1j * arr[:, 4])
        s21 = arr[:, 5] * np.exp(1j * arr[:, 6])
        s22 = arr[:, 7] * np.exp(1j * arr[:, 8])
        s = np.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


class Terminator(Model):
    """A terminator component that dissipates light into free space optimized
    for TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /_static/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the grating coupler. Must be either 'te' (default) or
        'tm'.
    """

    ocount = 1

    def __init__(self, pol="te"):
        if pol not in ["te", "tm"]:
            raise ValueError("'pol' must be one of 'te' or 'tm'")
        self.pol = pol

        if pol == "te":
            self._datafile = (
                "ebeam_terminator_te1550/nanotaper_w1=500,w2=60,L=10_TE.sparam"
            )
        else:
            self._datafile = (
                "ebeam_terminator_tm1550/nanotaper_w1=500,w2=60,L=10_TM.sparam"
            )

    def s_params(self, wl):
        file = _resolve_source_filepath(self._datafile)
        header, data = load_sparams(file)

        f, s = None, None
        for (p_out, p_in), sdf in data.groupby(["port_out", "port_in"]):
            freq = sdf["freq"].values
            if f is None:
                f = freq
            else:
                if not np.allclose(f, freq):
                    raise ValueError("Frequency mismatch between arrays in datafile.")

            if s is None:
                s = np.zeros((len(f), 1, 1), dtype=np.complex128)

            s[:, p_out - 1, p_in - 1] = sdf["mag"].values * np.exp(
                1j * sdf["phase"].values
            )

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


class Waveguide(Model):
    """Model for an waveguide optimized for TE polarized light at 1550
    nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /_static/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    pol : str, optional
        Polarization of the grating coupler. Must be either 'te' (default) or
        'tm'.
    length : float, optional
        Waveguide length in microns (default 0).
    width : float, optional
        Waveguide width in nanometers (default 500).
    height : float, optional
        Waveguide height in nanometers (default 220).
    loss : float, optional
        Loss of the waveguide in dB/cm (default 0).
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

    Sorted matrix of valid parameter sets for adiabatic tapers:

    ========  =======
    height    width
    ========  =======
        210      400
        210      420
        210      440
        210      460
        210      480
        210      500
        210      520
        210      540
        210      560
        210      580
        210      600
        210      640
        210      680
        210      720
        210      760
        210      800
        210      840
        210      880
        210      920
        210      960
        210     1000
        210     1040
        210     1080
        210     1120
        210     1160
        210     1200
        210     1240
        210     1280
        210     1320
        210     1360
        210     1400
        210     1500
        210     1600
        210     1700
        210     1800
        210     1900
        210     2000
        210     2100
        210     2200
        210     2300
        210     2400
        210     2500
        210     2600
        210     2700
        210     2800
        210     2900
        210     3000
        210     3100
        210     3200
        210     3300
        210     3400
        210     3500
        220      400
        220      420
        220      440
        220      460
        220      480
        220      500
        220      520
        220      540
        220      560
        220      580
        220      600
        220      640
        220      680
        220      720
        220      760
        220      800
        220      840
        220      880
        220      920
        220      960
        220     1000
        220     1040
        220     1080
        220     1120
        220     1160
        220     1200
        220     1240
        220     1280
        220     1320
        220     1360
        220     1400
        220     1500
        220     1600
        220     1700
        220     1800
        220     1900
        220     2000
        220     2100
        220     2200
        220     2300
        220     2400
        220     2500
        220     2600
        220     2700
        220     2800
        220     2900
        220     3000
        220     3100
        220     3200
        220     3300
        220     3400
        220     3500
        230      400
        230      420
        230      440
        230      460
        230      480
        230      500
        230      520
        230      540
        230      560
        230      580
        230      600
        230      640
        230      680
        230      720
        230      760
        230      800
        230      840
        230      880
        230      920
        230      960
        230     1000
        230     1040
        230     1080
        230     1120
        230     1160
        230     1200
        230     1240
        230     1280
        230     1320
        230     1360
        230     1400
        230     1500
        230     1600
        230     1700
        230     1800
        230     1900
        230     2000
        230     2100
        230     2200
        230     2300
        230     2400
        230     2500
        230     2600
        230     2700
        230     2800
        230     2900
        230     3000
        230     3100
        230     3200
        230     3300
        230     3400
        230     3500
    ========  =======
    """

    ocount = 2

    def __init__(
        self,
        pol="te",
        length=0,
        width=500,
        height=220,
        loss=0,
        sigma_ne=0.05,
        sigma_ng=0.05,
        sigma_nd=0.0001,
    ):
        if pol not in ["te", "tm"]:
            raise ValueError("Invalid polarization, must be either 'te' or 'tm'")

        df = self._generate_parameter_sets()
        if not ((df["width"] == width) & (df["height"] == height)).any():
            raise ValueError(
                "Invalid parameter set, see the documentation for valid parameter sets"
            )

        self.pol = pol
        self.length = length
        self.width = width
        self.height = height
        self.loss = loss
        self.sigma_ne = sigma_ne
        self.sigma_ng = sigma_ng
        self.sigma_nd = sigma_nd

        self._datafile = (
            f"wg_integral_source/WaveGuideTETMStrip,w={width},h={height}.txt"
        )

    @classmethod
    def _generate_parameter_sets(cls) -> pd.DataFrame:
        """Generate a dataframe of all valid parameter sets by parsing the
        filenames of the data files in the source directory."""
        pattern = r"WaveGuideTETMStrip,w=(?P<width>\d+),h=(?P<height>\d+).txt"
        path = _resolve_source_filepath("wg_integral_source")

        params = []
        for file in [str(p.name) for p in path.glob("*.txt")]:
            m = re.match(pattern, file)
            if m:
                params.append(m.groupdict())

        df = pd.DataFrame(params)
        df["width"] = pd.to_numeric(df["width"], errors="coerce")
        df["height"] = pd.to_numeric(df["height"], errors="coerce")

        # Find out which columns have the most unique items
        unique_counts = df.nunique()
        # Sort dataframe by fewest unique items in column to most unique items
        sorted_values = unique_counts.sort_values().index.tolist()
        df = df.sort_values(by=sorted_values, ignore_index=True)
        # Rearrange columns of dataframe to go from fewest unique items to most
        sorted_df = df[sorted_values]

        return sorted_df

    @classmethod
    def _generate_parameter_table_rst(cls) -> str:
        """Generate a rst table of valid parameter sets for the docstring."""
        df = cls._generate_parameter_sets()
        df.reset_index(drop=True, inplace=True)
        return tabulate(df.values, df.columns, tablefmt="rst")

    # TODO: Is this correct?
    def s_params(self, wl: Union[float, np.ndarray]):
        # Load data file, extract coefficients
        path = _resolve_source_filepath(self._datafile)
        arr = np.loadtxt(path)
        if self.pol == "te":
            lam0, ne, _, ng, _, nd, _ = arr
        else:  # tm
            lam0, _, ne, _, ng, _, nd = arr

        wl = np.asarray(wl).reshape(-1) * 1e-6  # convert microns to meters
        freqs = wl2freq(wl)  # convert wavelengths to freqs
        length = self.length * 1e-6  # convert microns to meters
        loss = self.loss * 100  # convert loss from dB/cm to dB/m

        # Initialize array to hold s-params
        s = np.zeros((len(freqs), 2, 2), dtype=np.complex128)

        # Loss calculation (convert loss to dB/m)
        alpha = loss / (20 * np.log10(np.exp(1)))
        w = 2 * np.pi * np.asarray(freqs)  # get angular freqs from freqs
        w0 = (2 * np.pi * SPEED_OF_LIGHT) / lam0  # center freqs (angular)

        # calculation of K
        K = (
            2 * np.pi * ne / lam0
            + (ng / SPEED_OF_LIGHT) * (w - w0)
            - (nd * lam0**2 / (4 * np.pi * SPEED_OF_LIGHT)) * ((w - w0) ** 2)
        )

        # build s-matrix from K and waveguide length
        s[:, 0, 1] = s[:, 1, 0] = np.exp(-alpha * length + (1j * K * length))
        return s


# TODO: Make sure the width parameter is accurately documented
class YBranch(Model):
    """SiEPIC EBeam PDK Y-branch model.

    A y-branch efficiently splits the input 50/50 between the two outputs.
    It can also be used as a combiner if used in the opposite direction,
    combining and interfering the light from two inputs into the one output.

    .. image:: /_static/images/ebeam_y_1550.png
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

        self._datafile = f"y_branch_source/Ybranch_Thickness ={self.thickness} width={self.width}.sparam"

    def s_params(self, wl):
        # Read from file
        file = _resolve_source_filepath(self._datafile)
        header, data = load_sparams(file)

        MODE_ID = self._POL_MAPPING[self.pol]

        # Build the s-matrix
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

        wl = np.array(wl) * 1e-6  # convert microns to meters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ntwk = skrf.Network(f=f, s=s, f_unit="Hz")
            interp_freq = Frequency.from_f(wl2freq(wl), unit="Hz")
            return ntwk.interpolate(interp_freq).s


# class BidirectionalCoupler(SiEPIC_PDK_Base):

#     def __init__(
#         self,
#         thickness=220e-9,
#         width=500e-9,
#         **kwargs,
#     ):
#         super().__init__(
#             **kwargs,
#             thickness=thickness,
#             width=width,
#         )

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.coupler(
#                 gap=0.2, length=50.55, dy=4.7, width=width * 1e6
#             )
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()
#                 widths = []
#                 heights = []
#                 s_params = []
#                 for idx in range(0, len(available), 2):
#                     d = available[idx]
#                     widths.append(d["width"])
#                     heights.append(d["thickness"])
#                     valid_args = available[idx]
#                     sparams = parser.read_params(self._get_file(valid_args))
#                     self._f, s = parser.build_matrix(sparams)
#                     s_params.append(s)

#                 s_params = np.asarray(s_params)

#                 widths = np.asarray(widths, dtype=np.double)
#                 heights = np.asarray(heights, dtype=np.double)

#                 dim = len(s_params)
#                 s_list = [s_params[dimidx][:][:][:] for dimidx in range(dim)]
#                 s_list = np.asarray(s_list, dtype=complex)
#                 self._s = interp.griddata(
#                     (widths, heights),
#                     s_list,
#                     (self.width * 1e9, self.thickness * 1e9),
#                     method="cubic",
#                 )

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()
#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()}
#                 for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             sparams = parser.read_params(self._get_file(valid_args))
#             self._f, self._s = parser.build_matrix(sparams)

#             self.freq_range = (self._f[0], self._f[-1])
#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)

#             self.enable_autoupdate()

#     def s_parameters(self, freqs):
#         return interpolate(freqs, self._f, self._s)

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_thickness = self.thickness

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.thickness = t

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.width = self.nominal_width
#         self.thickness = self.nominal_thickness


# class HalfRing(SiEPIC_PDK_Base):
#     def __init__(
#         self,
#         gap=30e-9,
#         radius=10e-6,
#         width=500e-9,
#         thickness=220e-9,
#         couple_length=0.0,
#         **kwargs,
#     ):
#         super().__init__(
#             **kwargs,
#             gap=gap,
#             radius=radius,
#             width=width,
#             thickness=thickness,
#             couple_length=couple_length,
#         )

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.coupler_ring(
#                 gap=gap * 1e6,
#                 length_x=couple_length * 1e6,
#                 radius=radius * 1e6,
#                 width=width * 1e6,
#             )
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()
#                 widths = []
#                 heights = []
#                 s_params = []
#                 for idx in range(0, len(available), 2):
#                     d = available[idx]
#                     widths.append(d["width"])
#                     heights.append(d["thickness"])
#                     valid_args = available[idx]
#                     sparams = parser.read_params(self._get_file(valid_args))
#                     sparams = list(
#                         filter(
#                             lambda sparams: sparams["mode"] == self.polarization,
#                             sparams,
#                         )
#                     )
#                     self._f, s = parser.build_matrix(sparams)

#                     s_params.append(s)

#                 s_params = np.asarray(s_params)

#                 widths = np.asarray(widths, dtype=complex)
#                 heights = np.asarray(heights, dtype=complex)

#                 dim, _, _, _ = s_params.shape
#                 s_list = []

#                 for dimidx in range(dim):
#                     s_list.append(s_params[dimidx][:][:][:])
#                 s_list = np.asarray(s_list, dtype=complex)
#                 self._s = interp.griddata(
#                     (widths, heights),
#                     s_list,
#                     (self.width * 1e9, self.thickness * 1e9),
#                     method="cubic",
#                 )

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()
#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v), 15) for k, v in d.items()} for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             sparams = parser.read_params(self._get_file(valid_args))
#             self._f, self._s = parser.build_matrix(sparams)

#             self.freq_range = (self._f[0], self._f[-1])
#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)

#             self.enable_autoupdate()

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_thickness = self.thickness

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.thickness = t

#     def s_parameters(self, freqs):
#         return interpolate(freqs, self._f, self._s)


# class DirectionalCoupler(SiEPIC_PDK_Base):
#     def __init__(self, gap=200e-9, Lc=10e-6, **kwargs):
#         super().__init__(
#             **kwargs,
#             gap=gap,
#             Lc=Lc,
#         )

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.coupler(
#                 gap=gap * 1e6, length=Lc * 1e6, dy=10
#             )
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()
#                 Lcs = []
#                 s_params = []
#                 for idx in range(len(available)):
#                     d = available[idx]
#                     Lcs.append(d["Lc"])
#                     valid_args = available[idx]
#                     sparams = parser.read_params(self._get_file(valid_args))
#                     self._f, s = parser.build_matrix(sparams)

#                     s_params.append(s)

#                 s_params = np.asarray(s_params)

#                 Lcs = [Lcs[i].replace("u", "") for i in range(len(Lcs))]
#                 Lcs = np.asarray(Lcs, dtype=float)

#                 dim, freq, inp, out = s_params.shape
#                 s_new = np.ndarray((freq, inp, out), dtype=complex)
#                 for freqidx in range(freq):
#                     for inpidx in range(inp):
#                         for outidx in range(out):
#                             s_list = []
#                             for dimidx in range(dim):
#                                 s_list.append(s_params[dimidx][freqidx][inpidx][outidx])

#                             s_interp = interp.interp1d(
#                                 Lcs, np.asarray(s_list), kind="cubic"
#                             )
#                             s_new[freqidx][inpidx][outidx] = s_interp(self.Lc * 1e6)

#                 self._s = s_new
#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()
#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v), 15) for k, v in d.items()} for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             sparams = parser.read_params(self._get_file(valid_args))
#             self._f, self._s = parser.build_matrix(sparams)

#             self.freq_range = (self._f[0], self._f[-1])
#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)

#             self.enable_autoupdate()

#     def update_variations(self, **kwargs):
#         self.nominal_Lc = self.Lc

#         w = self.nominal_Lc + kwargs.get("corr_w") * 1e-6

#         self.layout_aware = True
#         self.Lc = w

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.Lc = self.nominal_Lc


# class Terminator(SiEPIC_PDK_Base):

#     def __init__(
#         self,
#         w1=500e-9,
#         w2=60e-9,
#         L=10e-6,
#         **kwargs,
#     ):
#         super().__init__(
#             **kwargs,
#             w1=w1,
#             w2=w2,
#             L=L,
#         )

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.taper(
#                 width1=w1 * 1e6, width2=w2 * 1e6, with_two_ports=False
#             )
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))

#     def on_args_changed(self):
#         self.suspend_autoupdate()

#         available = self._source_argsets()
#         normalized = []
#         for d in available:
#             w1, w2, L = ((key, d.get(key)) for key in self._args_keys)
#             normalized.append(
#                 {
#                     w1[0]: round(str2float(w1[1]) * 1e-9, 15),
#                     w2[0]: round(str2float(w2[1]) * 1e-9, 15),
#                     L[0]: round(str2float(L[1]) * 1e-6, 15),
#                 }
#             )
#         idx = self._get_matched_args(normalized, self.args)

#         valid_args = available[idx]
#         sparams = parser.read_params(self._get_file(valid_args))
#         self._f, self._s = parser.build_matrix(sparams)

#         self.freq_range = (self._f[0], self._f[-1])
#         for key, value in normalized[idx].items():
#             setattr(self, key, value)

#         self.enable_autoupdate()


# class GratingCoupler(SiEPIC_PDK_Base):

#     def __init__(self):

#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.components.grating_coupler_te()
#             self.component.name = self.name
#             pin_names = [pin.name for pin in self.pins]
#             for i, port in enumerate(self.component.ports.values()):
#                 port.name = pin_names[i]
#             self.component.ports = dict(zip(pin_names, self.component.ports.values()))
#             port1 = self.component.ports["pin1"]
#             self.component.ports["pin1"] = self.component.ports["pin2"]
#             self.component.ports["pin2"] = port1

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 self._update_lay_aware_true()

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()
#         except AttributeError:
#             self.suspend_autoupdate()

#             self._update_no_lay_aware()

#             self.enable_autoupdate()

#     def _update_no_lay_aware(self):
#         available = self._source_argsets()
#         normalized = []
#         for d in available:
#             polarization, thickness, deltaw = (
#                 (key, d.get(key)) for key in self._args_keys
#             )
#             normalized.append(
#                 {
#                     polarization[0]: polarization[1],
#                     thickness[0]: round(str2float(thickness[1]) * 1e-9, 15),
#                     deltaw[0]: round(str2float(deltaw[1]) * 1e-9, 15),
#                 }
#             )
#         idx = self._get_matched_args(normalized, self.args)
#         for key, value in normalized[idx].items():
#             setattr(self, key, value)

#         valid_args = available[idx]
#         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         self._f = params[:, 0]
#         self._s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         self._s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         self._s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         self._s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         self._s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         # Arrays are from high frequency to low; reverse it,
#         # for convention's sake.
#         self._f = self._f[::-1]
#         self._s = self._s[::-1]
#         self.freq_range = (self._f[0], self._f[-1])

#     def _update_lay_aware_true(self):
#         pass
#         # available = self._source_argsets()
#         # thicknesses = []
#         # deltaws = []
#         # s_params = []
#         # for d in available:
#         #     _, thickness, deltaw = [(key, d.get(key)) for key in self._args_keys]
#         #     thicknesses.append(round(str2float(thickness[1]) * 1e-9, 15))
#         #     deltaws.append(round(str2float(deltaw[1]) * 1e-9, 15))

#         # if self.polarization == "TE":
#         #     for idx in range(round(len(thicknesses) / 2)):
#         #         valid_args = available[idx]
#         #         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         #         self._f = params[:, 0]
#         #         s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         #         s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         #         s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         #         s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         #         s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         #         self._f = self._f[::-1]
#         #         s = s[::-1]
#         #         s_params.append(s)

#         #     s_params = np.asarray(s_params, dtype=object)
#         #     thicknesses = np.asarray(thicknesses, dtype=float)
#         #     deltaws = np.asarray(deltaws, dtype=float)

#         #     dim = len(s_params)
#         #     for dimidx in range(dim):
#         #         print(f'[s_params[dimidx][:][:][:])
#         #         s_list.append(s_params[dimidx][:][:][:])
#         #     s_list = np.asarray(s_list, dtype=complex)
#         #     self._s = interp.griddata(
#         #         (
#         #             thicknesses[0 : round(len(thicknesses) / 2)],
#         #             deltaws[0 : round(len(deltaws) / 2)],
#         #         ),
#         #         s_list,
#         #         (self.thickness, self.deltaw),
#         #         method="cubic",
#         #     )

#         # elif self.polarization == "TM":
#         #     for idx in range(round(len(thicknesses) / 2) + 1, len(thicknesses)):
#         #         valid_args = available[idx]
#         #         params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
#         #         self._f = params[:, 0]
#         #         s = np.zeros((len(self._f), 2, 2), dtype="complex128")
#         #         s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
#         #         s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
#         #         s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
#         #         s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

#         #         self._f = self._f[::-1]
#         #         s = s[::-1]
#         #         s_params.append(s)

#         #     s_params = np.asarray(s_params, dtype=object)
#         #     thicknesses = np.asarray(thicknesses, dtype=float)
#         #     deltaws = np.asarray(deltaws, dtype=float)

#         #     dim = len(s_params)
#         #     s_list = []
#         #     for dimidx in range(dim):
#         #         s_list.append(s_params[dimidx][:][:][:])
#         #     s_list = np.asarray(s_list, dtype=complex)
#         #     self._s = interp.griddata(
#         #         (
#         #             thicknesses[round(len(thicknesses) / 2) + 1, len(thicknesses)],
#         #             deltaws[round(len(thicknesses) / 2) + 1, len(thicknesses)],
#         #         ),
#         #         s_list,
#         #         (self.thickness, self.deltaw),
#         #         method="cubic",
#         #     )

#     def update_variations(self, **kwargs):
#         self.nominal_deltaw = self.deltaw
#         self.nominal_thickness = self.thickness

#         w = self.deltaw + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.deltaw = w
#         self.thickness = t

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.thickness = self.nominal_thickness
#         self.deltaw = self.nominal_deltaw


# class Waveguide(SiEPIC_PDK_Base):

#     def __init__(
#         sigma_ne=0.05,
#         sigma_ng=0.05,
#         sigma_nd=0.0001,
#         **kwargs,
#     ):

#         if _has_gf:
#             self.path: Route = None

#         self.regenerate_monte_carlo_parameters()

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()

#                 widths = []
#                 heights = []
#                 for d in available:
#                     widths.append(d["width"])
#                     heights.append(d["height"])

#                 lam0_all = []
#                 ne_all = []
#                 ng_all = []
#                 nd_all = []
#                 for idx in range(len(available)):
#                     valid_args = available[idx]
#                     with open(self._get_file(valid_args)) as f:
#                         params = f.read().rstrip("\n")
#                     if self.polarization == "TE":
#                         lam0, ne, _, ng, _, nd, _ = params.split(" ")
#                     elif self.polarization == "TM":
#                         lam0, _, ne, _, ng, _, nd = params.split(" ")
#                         raise NotImplementedError

#                     lam0_all.append(lam0)
#                     ne_all.append(ne)
#                     ng_all.append(ng)
#                     nd_all.append(nd)

#                 widths = np.asarray(widths).astype(float)
#                 heights = np.asarray(heights).astype(float)
#                 lam0_all = np.asarray(lam0_all).astype(float)
#                 ne_all = np.asarray(ne_all).astype(float)
#                 ng_all = np.asarray(ng_all).astype(float)
#                 nd_all = np.asarray(nd_all).astype(float)

#                 self.lam0 = interp.griddata(
#                     (widths, heights),
#                     lam0_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.ne = interp.griddata(
#                     (widths, heights),
#                     ne_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.ng = interp.griddata(
#                     (widths, heights),
#                     ng_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )
#                 self.nd = interp.griddata(
#                     (widths, heights),
#                     nd_all,
#                     (self.width * 1e9, self.height * 1e9),
#                     method="cubic",
#                 )

#                 self.enable_autoupdate()

#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()}
#                 for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             with open(self._get_file(valid_args)) as f:
#                 params = f.read().rstrip("\n")
#             if self.polarization == "TE":
#                 lam0, ne, _, ng, _, nd, _ = params.split(" ")
#             elif self.polarization == "TM":
#                 lam0, _, ne, _, ng, _, nd = params.split(" ")
#                 raise NotImplementedError
#             self.lam0 = float(lam0)
#             self.ne = float(ne)
#             self.ng = float(ng)
#             self.nd = float(nd)

#             # Updates parameters width and thickness to closest match.
#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)

#             self.enable_autoupdate()

#     def s_parameters(self, freqs):
#         """Get the s-parameters of a waveguide.

#         Parameters
#         ----------
#         freqs : float
#             The array of frequencies to get s parameters for.

#         Returns
#         -------
#         (freqs, s) : tuple
#             Returns a tuple containing the frequency array, `freqs`,
#             corresponding to the calculated s-parameter matrix, `s`.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.ne, self.ng, self.nd
#         )

#     def monte_carlo_s_parameters(self, freqs):
#         """Returns a monte carlo (randomized) set of s-parameters.

#         In this implementation of the monte carlo routine, random values
#         are generated for ne, ng, and nd for each run through of the
#         monte carlo simulation. This means that all waveguide elements
#         throughout a single circuit will have the same (random) ne, ng,
#         and nd values. Hence, there is correlated randomness in the
#         monte carlo parameters but they are consistent within a single
#         circuit.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.rand_ne, self.rand_ng, self.rand_nd
#         )

#     def layout_aware_monte_carlo_s_parameters(self, freqs):
#         """Returns a monte carlo (randomized) set of s-parameters.

#         In this implementation of the monte carlo routine, values
#         generated for lam0, ne, ng, and nd using the Reduced Spatial
#         Correlation Matrix method are used to return a set of
#         s-parameters. This is repeated for each run of the Monte Carlo
#         analysis for every wavehuide component in the circuit.
#         """
#         return self.calc_s_params(
#             freqs, self.length, self.lam0, self.ne, self.ng, self.nd
#         )

#     def regenerate_monte_carlo_parameters(self):
#         self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
#         self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
#         self.rand_nd = np.random.normal(self.nd, self.sigma_nd)

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_height = self.height

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         h = self.height + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.height = h

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.width = self.nominal_width
#         self.height = self.nominal_height

#     @staticmethod
#     def calc_s_params(freqs, length, lam0, ne, ng, nd):
#         # Initialize array to hold s-params
#         s = np.zeros((len(freqs), 2, 2), dtype=complex)

#         # Loss calculation
#         TE_loss = 700  # dB/m for width 500nm
#         alpha = TE_loss / (20 * np.log10(np.exp(1)))

#         w = np.asarray(freqs) * 2 * np.pi  # get angular freqs from freqs
#         w0 = (2 * np.pi * SPEED_OF_LIGHT) / lam0  # center freqs (angular)

#         # calculation of K
#         K = (
#             2 * np.pi * ne / lam0
#             + (ng / SPEED_OF_LIGHT) * (w - w0)
#             - (nd * lam0**2 / (4 * np.pi * SPEED_OF_LIGHT)) * ((w - w0) ** 2)
#         )

#         for x in range(0, len(freqs)):  # build s-matrix from K and waveguide length
#             s[x, 0, 1] = s[x, 1, 0] = np.exp(-alpha * length + (K[x] * length * 1j))

#         return s


# class YBranch(SiEPIC_PDK_Base):

#     def __init__(self):
#         if _has_gf:
#             gf.clear_cache()
#             self.component = gf.read.import_gds(
#                 os.path.join(os.path.dirname(__file__), "source_data/ebeam_y_1550.gds")
#             )
#             self.component.name = self.name
#             self.component.ports[self.pins[0].name] = gf.Port(
#                 self.pins[0].name,
#                 180,
#                 center=(-7.4, 0),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )
#             self.component.ports[self.pins[1].name] = gf.Port(
#                 self.pins[1].name,
#                 0,
#                 center=(7.4, 2.75),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )
#             self.component.ports[self.pins[2].name] = gf.Port(
#                 self.pins[2].name,
#                 0,
#                 center=(7.4, -2.75),
#                 width=width * 1e6,
#                 layer="PORT",
#                 parent=self.component,
#             )

#     def on_args_changed(self):
#         try:
#             if self.layout_aware:
#                 self.suspend_autoupdate()

#                 available = self._source_argsets()
#                 widths = []
#                 heights = []
#                 s_params = []
#                 for idx in range(0, len(available)):
#                     d = available[idx]
#                     widths.append(d["width"])
#                     heights.append(d["thickness"])
#                     valid_args = available[idx]
#                     sparams = parser.read_params(self._get_file(valid_args))
#                     sparams = list(
#                         filter(
#                             lambda sparams: sparams["mode"] == self.polarization,
#                             sparams,
#                         )
#                     )
#                     self._f, s = parser.build_matrix(sparams)

#                     s_params.append(s)

#                 s_params = np.asarray(s_params)

#                 widths = np.asarray(widths, dtype=float)
#                 heights = np.asarray(heights, dtype=float)

#                 dim, _, _, _ = s_params.shape
#                 s_list = []
#                 for dimidx in range(dim):
#                     s_list.append(s_params[dimidx][:][:][:])
#                 s_list = np.asarray(s_list, dtype=complex)
#                 self._s = interp.griddata(
#                     (widths, heights),
#                     s_list,
#                     (self.width * 1e9, self.thickness * 1e9),
#                     method="cubic",
#                 )

#                 self.freq_range = (self._f[0], self._f[-1])

#                 self.enable_autoupdate()

#         except AttributeError:
#             self.suspend_autoupdate()

#             available = self._source_argsets()
#             normalized = [
#                 {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()}
#                 for d in available
#             ]
#             idx = self._get_matched_args(normalized, self.args)

#             valid_args = available[idx]
#             sparams = parser.read_params(self._get_file(valid_args))
#             sparams = list(
#                 filter(lambda sparams: sparams["mode"] == self.polarization, sparams)
#             )

#             for key, value in normalized[idx].items():
#                 setattr(self, key, value)
#             self._f, self._s = parser.build_matrix(sparams)
#             self.freq_range = (self._f[0], self._f[-1])

#             self.enable_autoupdate()

#     def update_variations(self, **kwargs):
#         self.nominal_width = self.width
#         self.nominal_thickness = self.thickness

#         w = self.width + kwargs.get("corr_w") * 1e-9
#         t = self.thickness + kwargs.get("corr_t") * 1e-9

#         self.layout_aware = True
#         self.width = w
#         self.thickness = t

#     def regenerate_layout_aware_monte_carlo_parameters(self):
#         self.width = self.nominal_width
#         self.thickness = self.nominal_thickness
