# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""SiEPIC models compatible with SAX."""

import importlib.resources
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import sax
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c as SPEED_OF_LIGHT
from tabulate import tabulate

import simphony.libraries
from simphony.plugins.lumerical import df_to_sdict, load_sparams
from simphony.utils import freq2wl, resample, wl2freq

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


@lru_cache()
def _load_txt_cached(path: Union[Path, str]) -> np.ndarray:
    """Loads a text file from the source_data directory and caches it.

    Parameters
    ----------
    filename : str
        The name of the file to be loaded.

    Returns
    -------
    content : str
        The contents of the file.
    """
    return np.loadtxt(path)


def _create_sdict_from_df(wl: Union[float, ArrayLike], df: pd.DataFrame) -> sax.SDict:
    """Create an s-dictionary from a dataframe of s-parameters.

    Parameters
    ----------
    wl : float or ArrayLike
        Wavelengths to interpolate the s-parameters to (in microns).
    df : pandas.DataFrame
        A dataframe of s-parameters. Must have columns 'port_in', 'port_out',
        'freq' (in Hz), 'mag', and 'phase'.
    """
    wl_m = jnp.array(wl).reshape(-1) * 1e-6  # meters
    f, s = df_to_sdict(df)
    new_s = resample(wl2freq(wl_m), f, s)
    return new_s


def _generate_parameter_sets(
    pattern: str,
    path: Union[str, Path],
    ext: str = "sparam",
    columns: List[str] = [],
) -> pd.DataFrame:
    """Generate a dataframe of all valid parameter sets by parsing the
    filenames of the data files in the source directory.

    Parameters
    ----------
    pattern : str
        Regex pattern with named fields to match filenames against.
    path : str or Path
        Path to the source directory.
    ext : str, optional
        File extension to match (default 'sparam').
    columns : list of str
        List of columns to convert to numeric types. Should be a subset of the
        named fields in the regex pattern.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of valid parameter sets.
    """
    path = _resolve_source_filepath(path)

    params = []
    for file in [str(p.name) for p in path.glob(f"*.{ext}")]:
        m = re.match(pattern, file)
        if m:
            params.append(m.groupdict())

    df = pd.DataFrame(params)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Find out which columns have the most unique items
    unique_counts = df.nunique()
    # Sort dataframe by fewest unique items in column to most unique items
    sorted_values = unique_counts.sort_values().index.tolist()
    df = df.sort_values(by=sorted_values, ignore_index=True)
    # Rearrange columns of dataframe to go from fewest unique items to most
    sorted_df = df[sorted_values]

    return sorted_df


def _generate_parameter_table_rst(df) -> str:
    """Generate an rst (text) table of valid parameter combinations.

    Useful for adding to the docstring of various models.

    Examples
    --------
    >>> df = _generate_parameter_sets()
    >>> print(_generate_parameter_table_rst(df))
    """
    df.reset_index(drop=True, inplace=True)
    return tabulate(df.values, df.columns, tablefmt="rst")


def _stringify_float(val: float, max_precision: int = 1) -> str:
    """Convert a float to a string, with a maximum precision."""
    if val == int(val):
        return str(int(val))
    else:
        return f"{val:.{max_precision}f}"


def bidirectional_coupler(
    wl: Union[float, ArrayLike] = 1.55,
    thickness: float = 220,
    width: float = 500,
) -> sax.SDict:
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
    if thickness not in [210.0, 220.0, 230.0]:
        raise ValueError("'thickness' must be one of 210, 220, or 230")
    thickness = str(int(thickness))

    if width not in [480.0, 500.0, 520.0]:
        raise ValueError("'width' must be one of 480, 500, or 520")
    width = str(int(width))

    datafile = f"bdc_TE_source/bdc_Thickness ={thickness} width={width}.sparam"

    file = _resolve_source_filepath(datafile)
    header, data = load_sparams(file)

    return _create_sdict_from_df(wl, data)


def directional_coupler(
    wl: Union[float, ArrayLike] = 1.55,
    gap: float = 200,
    coupling_length: float = 10.0,
) -> sax.SDict:
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
    Sorted matrix of valid parameter combinations for directional couplers:

    =====  =================
    gap    coupling_length
    =====  =================
    200                  0
    200                2.5
    200                  5
    200                7.5
    200                 10
    200               12.5
    200                 15
    200               17.5
    200                 20
    200               22.5
    200                 25
    200               27.5
    200                 30
    200               32.5
    200                 35
    200               37.5
    200                 40
    200               42.5
    200                 45
    200               47.5
    =====  =================
    """
    # df = self._generate_parameter_sets()
    # if not ((df["gap"] == gap) & (df["coupling_length"] == coupling_length)).any():
    #     raise ValueError(
    #         "Invalid parameter set, see the documentation for valid parameter sets"
    #     )

    _datafile = f"ebeam_dc_te1550/dc_gap={int(gap)}nm_Lc={_stringify_float(coupling_length)}um.sparam"

    file = _resolve_source_filepath(_datafile)
    header, data = load_sparams(file)

    return _create_sdict_from_df(wl, data)


def _generate_parameter_sets_dc() -> pd.DataFrame:
    """Generate a dataframe of all valid parameter sets by parsing the
    filenames of the data files in the source directory."""
    return _generate_parameter_sets(
        r"dc_gap=(?P<gap>\d+)nm_Lc=(?P<coupling_length>\d+)um.sparam",
        "ebeam_dc_te1550",
        ext="sparam",
        columns=["gap", "coupling_length"],
    )


def half_ring(
    wl: Union[float, ArrayLike] = 1.55,
    pol: Literal["te", "tm"] = "te",
    gap: float = 50,
    radius: float = 5,
    width: float = 500,
    thickness: float = 220,
    coupling_length: float = 0,
) -> sax.SDict:
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
    Sorted matrix of valid parameter combinations for half rings:

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

    if pol not in ["te", "tm"]:
        raise ValueError("'pol' must be one of 'te' or 'tm'")

    # df = self._generate_parameter_sets()
    # if not (
    #     (df["pol"] == pol)
    #     & (df["gap"] == gap)
    #     & (df["radius"] == radius)
    #     & (df["width"] == width)
    #     & (df["thickness"] == thickness)
    #     & (df["coupling_length"] == coupling_length)
    # ).any():
    #     raise ValueError(
    #         "Invalid parameter set, see the documentation for valid parameter sets"
    #     )
    # # (df == a).all(1).any()

    _datafile = f"ebeam_dc_halfring_straight/{pol}_ebeam_dc_halfring_straight_gap={int(gap)}nm_radius={int(radius)}um_width={int(width)}nm_thickness={int(thickness)}nm_CoupleLength={int(coupling_length)}um.dat"

    file = _resolve_source_filepath(_datafile)
    header, data = load_sparams(file)

    return _create_sdict_from_df(wl, data)


def _generate_parameter_sets_half_ring() -> pd.DataFrame:
    """Generate a dataframe of all valid parameter sets by parsing the
    filenames of the data files in the source directory."""
    return _generate_parameter_sets(
        r"(?P<pol>[a-z]+)_ebeam_dc_halfring_straight_gap=(?P<gap>\d+)nm_radius=(?P<radius>\d+)um_width=(?P<width>\d+)nm_thickness=(?P<thickness>\d+)nm_CoupleLength=(?P<coupling_length>\d+)um.dat"
        "ebeam_dc_halfring_straight",
        ext="dat",
        columns=["gap", "radius", "width", "thickness", "coupling_length"],
    )


def taper(
    wl: Union[float, ArrayLike] = 1.55,
    w1: float = 0.5,
    w2: float = 1.0,
    length: float = 10.0,
) -> sax.SDict:
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
    Sorted matrix of valid parameter combinations for adiabatic tapers:

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
    # df = self._generate_parameter_sets()
    # if not ((df["w1"] == w1) & (df["w2"] == w2) & (df["length"] == length)).any():
    #     raise ValueError(
    #         "Invalid parameter set, see the documentation for valid parameter sets"
    #     )

    _datafile = (
        f"ebeam_taper_te1550/w1={w1:.1f}um_w2={int(w2)}um_length={int(length)}um.dat"
    )

    path = _resolve_source_filepath(_datafile)
    arr = _load_txt_cached(path)

    f = arr[:, 0]
    wl_samp = freq2wl(f)  # meters
    wl_des = jnp.asarray(wl).reshape(-1) * 1e-6  # meters

    s11 = jnp.interp(wl_des, wl_samp, arr[:, 1] * jnp.exp(1j * arr[:, 2]))
    s12 = jnp.interp(wl_des, wl_samp, arr[:, 3] * jnp.exp(1j * arr[:, 4]))
    s21 = jnp.interp(wl_des, wl_samp, arr[:, 5] * jnp.exp(1j * arr[:, 6]))
    s22 = jnp.interp(wl_des, wl_samp, arr[:, 7] * jnp.exp(1j * arr[:, 8]))

    sdict = {
        ("o0", "o0"): s11,
        ("o0", "o1"): s12,
        ("o1", "o0"): s21,
        ("o1", "o1"): s22,
    }

    return sdict


def _generate_parameter_sets_taper() -> pd.DataFrame:
    """Generate a dataframe of all valid parameter sets by parsing the
    filenames of the data files in the source directory."""
    return _generate_parameter_sets(
        r"w1=(?P<w1>\d+\.\d+)um_w2=(?P<w2>\d+)um_length=(?P<length>\d+)um.dat"
        "ebeam_taper_te1550",
        ext="dat",
        columns=["w1", "w2", "length"],
    )


def terminator(
    wl: Union[float, ArrayLike] = 1.55,
    pol: Literal["te", "tm"] = "te",
) -> sax.SDict:
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
    if pol not in ["te", "tm"]:
        raise ValueError("'pol' must be one of 'te' or 'tm'")

    if pol == "te":
        _datafile = "ebeam_terminator_te1550/nanotaper_w1=500,w2=60,L=10_TE.sparam"
    else:
        _datafile = "ebeam_terminator_tm1550/nanotaper_w1=500,w2=60,L=10_TM.sparam"

    file = _resolve_source_filepath(_datafile)
    header, data = load_sparams(file)

    return _create_sdict_from_df(wl, data)


def grating_coupler(
    wl: Union[float, Array] = 1.55,
    pol: Literal["te", "tm"] = "te",
    thickness: float = 220.0,
    dwidth: float = 0,
) -> sax.SDict:
    """SiEPIC EBeam PDK grating coupler optimized for TE polarizations at
    1550nm.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees
    [needs citation].

    .. image:: /_static/images/ebeam_gc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    wl : float or Array
        The wavelengths to evaluate at in microns.
    pol : {"te", "tm"}
        Polarization of the input/output modes.
    thickness : {210.0, 220.0, 230.0}
        Thickness of the grating coupler silicon in nm. Useful for simulating
        manufacturing variability.
    dwidth : {-20.0, 0.0, 20.0}
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
    if pol not in ["te", "tm"]:
        raise ValueError("'pol' must be either 'te' or 'tm'")
    pol = pol.upper()

    if thickness not in [210.0, 220.0, 230.0]:
        raise ValueError("'thickness' must be one of 210.0, 220.0, or 230.0")
    thickness = str(int(thickness))

    if dwidth not in [-20.0, 0.0, 20.0]:
        raise ValueError("'dwidth' must be one of -20, 0, or 20")
    dwidth = str(int(dwidth))

    _datafile = f"gc_source/GC_{pol}1550_thickness={thickness} deltaw={dwidth}.txt"
    path = _resolve_source_filepath(_datafile)
    arr = _load_txt_cached(path)

    f = arr[:, 0]
    wl_samp = freq2wl(f)  # meters
    wl_des = jnp.asarray(wl).reshape(-1) * 1e-6  # meters

    s11 = jnp.interp(wl_des, wl_samp, arr[:, 1] * jnp.exp(1j * arr[:, 2]))
    s12 = jnp.interp(wl_des, wl_samp, arr[:, 3] * jnp.exp(1j * arr[:, 4]))
    s21 = jnp.interp(wl_des, wl_samp, arr[:, 5] * jnp.exp(1j * arr[:, 6]))
    s22 = jnp.interp(wl_des, wl_samp, arr[:, 7] * jnp.exp(1j * arr[:, 8]))

    sdict = {
        ("o0", "o0"): s11,
        ("o0", "o1"): s12,
        ("o1", "o0"): s21,
        ("o1", "o1"): s22,
    }

    return sdict


def waveguide(
    wl: Union[float, Array] = 1.55,
    pol: Literal["te", "tm"] = "te",
    length: float = 0.0,
    width: float = 500.0,
    height: float = 220.0,
    loss: float = 0.0,
) -> sax.SDict:
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

    Sorted matrix of valid parameter combinations for waveguides:

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
    if pol not in ["te", "tm"]:
        raise ValueError("Invalid polarization, must be either 'te' or 'tm'")

    width, height = int(width), int(height)

    # df = self._generate_parameter_sets()
    # if not ((df["width"] == width) & (df["height"] == height)).any():
    #     raise ValueError(
    #         "Invalid parameter set, see the documentation for valid parameter sets"
    #     )

    _datafile = f"wg_integral_source/WaveGuideTETMStrip,w={width},h={height}.txt"

    # Load data file, extract coefficients
    path = _resolve_source_filepath(_datafile)
    arr = _load_txt_cached(path)

    if pol == "te":
        lam0, ne, _, ng, _, nd, _ = arr
    else:  # tm
        lam0, _, ne, _, ng, _, nd = arr

    wl_m = jnp.asarray(wl).reshape(-1) * 1e-6  # convert microns to meters
    freqs = wl2freq(wl_m)  # convert wavelengths to freqs
    length_m = length * 1e-6  # convert microns to meters

    loss = loss * 100  # convert loss from dB/cm to dB/m
    alpha = loss / (20 * jnp.log10(jnp.exp(1)))  # convert loss to m^-1
    omega = 2 * jnp.pi * jnp.asarray(freqs)  # get angular freqs from freqs
    omega0 = (2 * jnp.pi * SPEED_OF_LIGHT) / lam0  # center freqs (angular)

    # calculation of K
    K = (
        2 * jnp.pi * ne / lam0
        + (ng / SPEED_OF_LIGHT) * (omega - omega0)
        - (nd * lam0**2 / (4 * jnp.pi * SPEED_OF_LIGHT)) * ((omega - omega0) ** 2)
    )

    sdict = {
        ("o0", "o0"): jnp.zeros(wl_m.shape, dtype=np.complex128),
        ("o0", "o1"): jnp.exp(-alpha * length_m + (1j * K * length_m)),
        ("o1", "o0"): jnp.exp(-alpha * length_m + (1j * K * length_m)),
        ("o1", "o1"): jnp.zeros(wl_m.shape, dtype=np.complex128),
    }
    return sdict


def _generate_parameter_sets_waveguide() -> pd.DataFrame:
    """Generate a dataframe of all valid parameter sets by parsing the
    filenames of the data files in the source directory."""
    return _generate_parameter_sets(
        r"WaveGuideTETMStrip,w=(?P<width>\d+),h=(?P<height>\d+).txt",
        "wg_integral_source",
        ext="txt",
        columns=["width", "height"],
    )


def y_branch(
    wl: Union[float, Array] = 1.55,
    pol: Literal["te", "tm"] = "te",
    thickness: float = 220.0,
    width: float = 500.0,
) -> sax.SDict:
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
    if pol not in ["te", "tm"]:
        raise ValueError("'pol' must be one of 'te' or 'tm'")

    if thickness not in [210.0, 220.0, 230.0]:
        raise ValueError("'thickness' must be one of 210, 220, or 230")
    thickness = str(int(thickness))

    if width not in [480.0, 500.0, 520.0]:
        raise ValueError("'width' must be one of 480, 500, or 520")
    width = str(int(width))

    _datafile = f"y_branch_source/Ybranch_Thickness ={thickness} width={width}.sparam"

    file = _resolve_source_filepath(_datafile)
    header, data = load_sparams(file)

    _POL_MAPPING = {"te": 1, "tm": 2}
    MODE_ID = _POL_MAPPING[pol]
    data = data[data.mode_out == MODE_ID]

    return _create_sdict_from_df(wl, data)
