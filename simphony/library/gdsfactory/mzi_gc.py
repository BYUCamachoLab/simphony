import pp
from simphony.library import siepic
from simphony.library.gdsfactory import sweep_simulation
from simphony.library.gdsfactory.mmi1x2 import mmi1x2
from simphony.netlist import Subcircuit


@pp.autoname
def mzi_gc(L0=1, L1=100, L2=10, y_model_factory=mmi1x2):
    """ MZI with grating couplers
    Deprecated!
    use add_gc instead
    """
    y = pp.call_if_func(y_model_factory)
    gc = siepic.ebeam_gc_te1550()
    wg_long = siepic.ebeam_wg_integral_1550(length=(2 * L0 + 2 * L1 + L2) * 1e-6)
    wg_short = siepic.ebeam_wg_integral_1550(length=(2 * L0 + L2) * 1e-6)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("MZI")
    circuit.add(
        [
            (gc, "input"),
            (gc, "output"),
            (y, "splitter"),
            (y, "recombiner"),
            (wg_long, "wg_long"),
            (wg_short, "wg_short"),
        ]
    )

    # You can set pin names individually:
    circuit.elements["input"].pins["n2"] = "input"
    circuit.elements["output"].pins["n2"] = "output"

    # Or you can rename all the pins simultaneously:
    # circuit.elements["splitter"].pins = ("in1", "out1", "out2")
    # circuit.elements["recombiner"].pins = ("out1", "in2", "in1")

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("input", "n1", "splitter", "W0"),
            ("splitter", "E0", "wg_long", "n1"),
            ("splitter", "E1", "wg_short", "n1"),
            ("recombiner", "E0", "wg_long", "n2"),
            ("recombiner", "E1", "wg_short", "n2"),
            ("output", "n1", "recombiner", "W0"),
        ]
    )
    return circuit


if __name__ == "__main__":
    c = mzi_gc()
    sweep_simulation(c)
