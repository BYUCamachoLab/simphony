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

import numpy as np
import pandas as pd
from lark import Lark, Transformer, v_args

sparams_grammar = r"""
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


def destring(string: str) -> str:
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


class SparamsTransformer(Transformer):
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
    def port(self, port) -> int:
        port = destring(port)
        if port.startswith("port"):
            return int(port.split(" ")[1])
        else:
            raise ValueError(
                f"Port name '{port}' is not supported, contact the developers."
            )

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
        params = destring(str(params))
        return params.split(";")

    def MODE(self, args) -> str:
        return destring(str(args))

    def VALUETYPE(self, args) -> str:
        return destring(str(args))

    def INT(self, args) -> int:
        return int(args)

    def STRING(self, args) -> str:
        return str(args)


parser = Lark(
    sparams_grammar, start="start", parser="lalr", transformer=SparamsTransformer()
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

    Notes
    -----
    Scattering parameters are returned in a DataFrame where the input and
    output ports, input and output modes, frequency, magnitude and phase, and
    sweep parameter columns (if applicable) are all present.

    You can learn more about the Lumerical S-parameter file format here:
    https://optics.ansys.com/hc/en-us/articles/360036618513-S-parameter-file-formats
    """
    with open(filename) as f:
        text = f.read()
    return parser.parse(text)
