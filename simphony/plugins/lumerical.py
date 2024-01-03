# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""This module contains convenience functions for parsing Lumerical .sparam
data files.

It works with all formats with optional extras as described on their
`website <https://optics.ansys.com/hc/en-us/articles/360036618513-S-parameter-file-formats>`_.
Results are compiled into a pandas DataFrame that can be filtered and grouped
as needed to construct the desired s-parameter matrix.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import sax
from jax import Array
from jax.typing import ArrayLike
from lark import Lark, Transformer, v_args

from simphony.utils import wl2freq

_sparams_grammar = r"""
    ?start: _EOL* [header] datablock+ _EOL*

    header: option1 | option2
    option1: "[" INT "," INT "]" _EOL
    option2: ("[" port "," position "]" _EOL)+

    datablock: ports shape values

    ports: "(" port "," MODE "," modeid "," port "," modeid "," VALUETYPE ["," groupdelay] ["," sweepparams] ")" _EOL
    shape: "(" INT "," INT ")" _EOL
    values: row+

    row: (SIGNED_NUMBER)+ _EOL
    position: ("'" [SIDE] "'") | ("\"" [SIDE] "\"")
    port: STRING
    modeid: INT
    VALUETYPE: STRING
    groupdelay: NUMBER
    sweepparams: STRING

    SIDE: "TOP" | "BOTTOM" | "LEFT" | "RIGHT"
    MODE: ("'" [POL] "'") | ("\"" [POL] "\"") | STRING
    POL: "TE" | "TM"
    _EOL: NEWLINE
    STRING: ("'" /[^']*?/ _STRING_ESC_INNER "'") | ("\"" /[^\"]*?/ _STRING_ESC_INNER "\"")

    %import common._STRING_ESC_INNER
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.INT
    %import common.WS_INLINE
    %import common.WS
    %import common.NEWLINE
    %ignore WS_INLINE
    """


def _destring(string: str) -> str:
    """Removes all single and double quotes from a string.

    Parameters
    ----------
    string : str
        String to remove quotes from.

    Returns
    -------
    str
        String with all single and double quotes removed.
    """
    return string.replace("'", "").replace('"', "")


class _SparamsTransformer(Transformer):
    @v_args(inline=True)
    def start(self, header, *datablocks):
        data = pd.concat(datablocks, ignore_index=True)
        return header, data

    @v_args(inline=True)
    def datablock(self, ports, shape, values):
        sweepparams = ports["sweepparams"] or []
        columns = sweepparams + ["freq", "mag", "phase"]
        rows, cols = shape
        if cols == len(columns) + 1:
            columns += ["groupdelay"]
        df = pd.DataFrame(values, columns=columns)
        df.loc[:, "port_out"] = ports["port_out"]
        df.loc[:, "port_in"] = ports["port_in"]
        df.loc[:, "mode_out"] = ports["mode_out"]
        df.loc[:, "mode_in"] = ports["mode_in"]
        return df

    @v_args(inline=True)
    def ports(
        self,
        port_out,
        mode_type,
        mode_out,
        port_in,
        mode_in,
        valuetype,
        groupdelay,
        sweepparams,
    ):
        return {
            "port_out": port_out,
            "mode_type": mode_type,
            "mode_out": mode_out,
            "port_in": port_in,
            "mode_in": mode_in,
            "valuetype": valuetype,
            "groupdelay": groupdelay,
            "sweepparams": sweepparams,
        }

    def shape(self, args) -> Tuple[int, int]:
        return tuple([int(arg) for arg in args])

    def values(self, args) -> np.ndarray:
        return np.array(args)

    def row(self, args) -> np.ndarray:
        return np.array([float(arg) for arg in args])

    @v_args(inline=True)
    def port(self, port) -> str:
        return _destring(port)

    @v_args(inline=True)
    def modeid(self, mid) -> int:
        if isinstance(mid, int):
            return mid
        else:
            raise ValueError(
                f"Mode ID '{mid}' is not supported, contact the developers."
            )

    @v_args(inline=True)
    def sweepparams(self, params) -> List[str]:
        params = _destring(str(params))
        return params.split(";")

    def MODE(self, args) -> str:
        return _destring(str(args))

    def VALUETYPE(self, args) -> str:
        return _destring(str(args))

    def INT(self, args) -> int:
        return int(args)

    def STRING(self, args) -> str:
        return str(args)


_parser = Lark(
    _sparams_grammar, start="start", parser="lalr", transformer=_SparamsTransformer()
)


@lru_cache()
def load_sparams(filename: Union[Path, str]) -> dict:
    """Load S-parameters from a Lumerical ".sparam" or ".dat" file.

    Parameters
    ----------
    filename : Path or str
        Path to the file.

    Returns
    -------
    header, data : tuple of dict, pd.DataFrame
        Tuple, where the first item is the header information dictionary and
        the second item is a DataFrame with the s-parameters.

    See Also
    --------
    df_to_sdict : Create an s-dictionary from a dataframe of s-parameters.
    save_sparams : Exports scattering parameters to a ".sparam" file readable by interconnect.

    Notes
    -----
    Scattering parameters are returned in a DataFrame where the input and
    output ports, input and output modes, frequency, magnitude and phase, and
    sweep parameter columns (if applicable) are all present.

    You can learn more about the Lumerical S-parameter file format here:
    https://optics.ansys.com/hc/en-us/articles/360036618513-S-parameter-file-formats
    """
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return _parser.parse(text)


def save_sparams(
    sparams: ArrayLike,
    wavelength: ArrayLike,
    filename: Union[str, Path],
    overwrite: bool = True,
) -> None:
    """Exports scattering parameters to a ".sparam" file readable by
    interconnect.

    Parameters
    ----------
    sparams : ArrayLike
        Numpy array of size *(N, d, d)* where *N* is the number of frequency points and *d* the number of ports.
    wavelength : ArrayLike
        Array of wavelengths (in um) of size *N*.
    filename : str or Path
        File path to save file.
    overwrite : bool, optional
        If True (default), overwrites any existing file.
    """
    # TODO: Test this function.
    _, d, _ = sparams.shape
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise FileExistsError(
            f"{filename} already exists, set overwrite=True to overwrite."
        )
    elif filename.exists() and overwrite:
        filename.unlink()

    with open(filename, "a", encoding="utf-8") as file:
        # make frequencies
        freq = wl2freq(wavelength * 1e-6)

        # iterate through sparams saving
        for in_ in range(d):
            for out in range(d):
                # put things together
                sp = sparams[:, in_, out]
                temp = np.vstack((freq, np.abs(sp), np.unwrap(np.angle(sp)))).T

                # Save header
                header = f'("port {out + 1}", "TE", 1, "port {in_ + 1}", 1, "transmission")\n'
                header += f"{temp.shape}"

                # save data
                np.savetxt(file, temp, header=header, comments="")


def df_to_sdict(df: pd.DataFrame) -> Tuple[Array, sax.SDict]:
    """Create an s-dictionary from a dataframe of s-parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of s-parameters. Usually the output of ``load_sparams``.
        Expected columns are 'port_in', 'port_out', 'mode_in', 'mode_out',
        'freq' (in Hz), 'mag', and 'phase'.

    Returns
    -------
    f : numpy.ndarray
        Array of frequencies (in Hz).
    sdict : sax.SDict
        Dictionary of scattering parameters.

    See Also
    --------
    load_sparams : Load s-parameters from a Lumerical .sparam file.
    """
    df = df.copy()
    df = df.sort_values("freq")

    if df["mode_out"].unique().size == 1 or df["mode_in"].unique().size == 1:
        multimode = False
        grouper = ["port_out", "port_in"]
    else:
        multimode = True
        grouper = ["port_out", "port_in", "mode_out", "mode_in"]

    f = None
    sdict = {}
    for keys, sdf in df.groupby(grouper):
        if multimode:
            p_out, p_in, m_out, m_in = keys
        else:
            p_out, p_in = keys

        # Ensure frequencies are matched across arrays
        freq = sdf["freq"].values
        if f is None:
            f = freq
        else:
            if not jnp.allclose(f, freq):
                raise ValueError("Frequency mismatch between arrays in datafile.")

        snn = sdf["mag"].values * jnp.exp(1j * sdf["phase"].values)

        if multimode:
            sdict[(f"{p_out}@{m_out - 1}", f"{p_in}@{m_in - 1}")] = snn
        else:
            sdict[(p_out, p_in)] = snn

    f = jnp.array(f).reshape(-1)
    return f, sdict
