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
    ws          = ~"\s*"

    # word        = ~r"[-\w]+"
    # emptyline   = ws+
    """
)

# class Paramset:
#     def __init__(self, input_port, output_port, mode, type_, f, s):
#         self.input_port = input_port
#         self.output_port = output_port
#         self.mode = mode
#         self.f = np.array(f)
#         self.s = np.array(s)
#         self.type_ = type_

class ParamVisitor(NodeVisitor):
    def visit_file(self, node, visited_children):
        """
        Handles the grammar:
            file        = paramset* ws*
        
        Collects all paramsets from the file and returns them as a list.

        Returns:
            list : List of all paramsets in the file.
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
        header['f'] = f
        header['s'] = s
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
        _, input_port, _, mode, _, number, _, output_port, _, number, _, type_, *_ = visited_children
        return dict(input_port=input_port, output_port=output_port, mode=mode, type_=type_)

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
        return (freq, real * np.exp(1j*imag))

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
        """ The generic visit method. """
        return visited_children or node

# data_sparam = """["port 1",""]
# ["port 2",""]
# ('port 1','TE',1,'port 1',1,"transmission")
# (51,3)
# 1.8737e+14	0.0380561	-2.56957
# 1.8762e+14	0.0379328	-2.5491
# ('port 1',TE,1,'port 1',1,"transmission")
# (51,3)
# 1.8737e+14	0.0380561	-2.56957
# 1.8762e+14	0.0379328	-2.5491

# """

# tree = sparam_grammar.parse(data_sparam)
# pv = ParamVisitor()
# output = pv.visit(tree)
# print(output)

with open('Ybranch_Thickness =220 width=500.sparam', 'r') as f:
    tree = sparam_grammar.parse(f.read())
pv = ParamVisitor()
output = pv.visit(tree)
print(output)

# Thank you, Stack Overflow: https://stackoverflow.com/questions/8653516/python-list-of-dictionaries-search
te = list(filter(lambda paramset: paramset['mode'] == 'TE', output))

# with open('te_ebeam_dc_halfring_straight_gap=30nm_radius=3um_width=520nm_thickness=210nm_CoupleLength=0um.dat', 'r') as f:
#     tree = sparam_grammar.parse(f.read())
# pv = ParamVisitor()
# output = pv.visit(tree)
# print(output)
