# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.libraries.siepic.parser
=======================

The terminology `paramset` in this module is different from its use in the
rest of the SiEPIC library.
"""

# See here: https://github.com/erikrose/parsimonious

import numpy as np
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor

sparam_grammar = Grammar(
    r"""
    file        = preamble* paramset* ws*
    preamble    = lbrack quoted comma ws* quoted rbrack ws*
    paramset    = header shape datapoint* ws*
    header      = lpar port comma mode comma number comma port comma number comma type rpar ws*
    shape       = lpar number comma ws* number rpar ws*
    datapoint   = number ws+ number ws+ number ws*

    port        = quote "port" ws number quote
    mode        = quote? string quote?
    type        = quote? string quote?

    string      = ~r"[-\s\w\d]+"
    number      = ~r"[-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?"
    quoted      = ~'"[^\"]*"|\'[^\']*\''
    quote       = ~r"\"|\'"
    lpar        = "("
    rpar        = ")"
    lbrack      = "["
    rbrack      = "]"
    comma       = ","
    ws          = ~r"\s*"
    """
)


class ParamVisitor(NodeVisitor):
    def visit_file(self, node, visited_children):
        """
        Handles the grammar:
            file        = paramset* ws*

        Collects all paramsets from the file and returns them as a list.

        Returns:
            list : List of all paramsets (dict) in the file.
        """
        paramsets = []
        for paramset in visited_children[1]:
            paramsets.append(paramset)
        return paramsets

    def visit_paramset(self, node, visited_children):
        """
        Handles the grammar:
            paramset    = header shape datapoint* ws*

        Example:
            ('port 1','TE',1,'port 1',1,"transmission")
            (51,3)
            1.8737e+14	0.0380561	-2.56957
            1.8762e+14	0.0379328	-2.5491

        Returns:
            dict : an initialized dictionary object.
        """
        header, shape, datapoints, _ = visited_children
        f = []
        s = []
        for child in datapoints:
            f.append(child[0])
            s.append(child[1])
        header["f"] = f
        header["s"] = s
        return header

    def visit_header(self, node, visited_children):
        """
        Handles the grammar:
            header      = lpar port comma mode comma number comma port comma number comma type rpar ws*

        Example:
            ("port 1","mode 1",1,"port 1",1,"transmission")
                ^                   ^
              input               output

        Returns:
            dict : keys=(input_port, output_port, mode, type_)
        """
        # FIXME: Investigate the order of output_port, input_port to make sure
        # this is correct.
        (
            _,
            output_port,
            _,
            mode,
            _,
            number,
            _,
            input_port,
            _,
            number,
            _,
            type_,
            *_,
        ) = visited_children
        return dict(
            input_port=input_port, output_port=output_port, mode=mode, type_=type_
        )

    def visit_datapoint(self, node, visited_children):
        """
        Handles the grammar:
            datapoint   = number ws+ number ws+ number ws*

        Example:
            1.8737028625000000e+14 8.8032014721294136e-04 -4.9073858469422826e-01

        Returns:
            (float, np.complex128) : tuple of frequency value and complex transmission
        """
        freq, _, real, _, imag, *_ = node.children
        freq, real, imag = float(freq.text), float(real.text), float(imag.text)
        return (freq, real * np.exp(1j * imag))

    def visit_port(self, node, visited_children):
        """
        Handles the grammar:
            port        = quote "port" ws number quote

        Example:
            "port 1"

        Returns:
            int : the port number alone, cast to an int
        """
        _, _, _, number, _ = visited_children
        return int(number)

    def visit_mode(self, node, visited_children):
        """
        Handles the grammar:
            mode        = quote? string quote?

        Example:
            "transmission"

        Returns:
            str : the value without the quotation marks
        """
        _, mode, _ = visited_children
        return mode.text

    def visit_type(self, node, visited_children):
        """
        Handles the grammar:
            type        = quote? string quote?

        Example:
            "transmission"

        Returns:
            str : the value without the quotation marks
        """
        _, typ, _ = visited_children
        return typ.text

    def visit_number(self, node, visited_children):
        """
        Handles the grammar:
            number      = ~r"[-+]?[0-9]+[.]?[0-9]*([eE][-+]?[0-9]+)?"

        Example:
            1.8737028625000000e+14

        Returns:
            float : the number cast to a float.
        """
        value = float(node.text)
        return value

    def generic_visit(self, node, visited_children):
        """The generic visit method."""
        return visited_children or node


def read_params(filename):
    """
    Parameters
    ----------
    filename : str
        Absolute path to file to be parsed.

    Returns
    -------
    list of dict
        Returns a list of dictionaries as constructed by ParamVisitor.
        Dictionary contains frequency array, s-parameters, and other
        information on a port-by-port basis.
    """
    with open(filename, "r") as f:
        tree = sparam_grammar.parse(f.read())
    pv = ParamVisitor()
    return pv.visit(tree)


def build_matrix(dicts):
    """Builds an s-parameter matrix and frequency array from a list of
    dictionaries generated by the parser.

    Parameters
    ----------
    dicts : list of dict
        The list of dictionaries that comprise one (and only one) model and
        its corresponding s-matrix. The dictionaries contain port and
        s-parameter information.

    Returns
    -------
    f, s : tuple(np.ndarray, np.ndarray)
        The first value is an array with all frequency values stored in it. The
        second value is a 3-dimensional matrix with the s-parameters for port-
        to-port interactions indexed by frequency.
    """
    f = dicts[0]["f"]
    shape = int(np.sqrt(len(dicts)))
    s = np.zeros((len(f), shape, shape), dtype="complex128")
    for d in dicts:
        if not np.array_equal(f, d["f"]):
            raise ValueError(
                "Frequency arrays for the same model do not match (invalid model)!"
            )
        s[:, d["output_port"] - 1, d["input_port"] - 1] = d["s"]
    return f, s
