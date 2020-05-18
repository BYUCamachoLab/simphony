import pp
from simphony.library import siepic
from simphony.library.gdsfactory import sweep_simulation
from simphony.library.gdsfactory.mzi import mzi
from simphony.netlist import Subcircuit


def add_gc(circuit, gc=siepic.ebeam_gc_te1550):
    """ add input and output gratings

    Args:
        circuit: needs to have `input` and `output` pins
        gc: grating coupler
    """
    c = Subcircuit(f"{circuit}_gc")
    gc = pp.call_if_func(gc)
    c.add(
        [(gc, "gci"), (gc, "gco"), (circuit, "circuit"),]
    )
    c.connect_many(
        [("gci", "n1", "circuit", "input"), ("gco", "n1", "circuit", "output"),]
    )

    # c.elements["circuit"].pins["input"] = "input_circuit"
    # c.elements["circuit"].pins["output"] = "output_circuit"
    c.elements["gci"].pins["n2"] = "input"
    c.elements["gco"].pins["n2"] = "output"

    return c


if __name__ == "__main__":
    c1 = mzi()
    c2 = add_gc(c1)
    sweep_simulation(c2)
