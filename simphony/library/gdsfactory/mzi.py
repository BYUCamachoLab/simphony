import matplotlib.pyplot as plt
import pp
from simphony.library import siepic
from simphony.library.gdsfactory.mmi1x2 import mmi1x2
from simphony.netlist import Subcircuit
from simphony.simulation import MonteCarloSweepSimulation, SweepSimulation


def mzi_circuit(L0=1, L1=100, L2=10, y_model_factory=mmi1x2):
    """ Mzi

    Args:
        L0: vertical length for both and top arms
        L1: bottom arm extra length
        L2: L_top horizontal length

    .. code::

               __L2__
               |      |
               L0     L0r
               |      |
     splitter==|      |==recombiner
               |      |
               L0     L0r
               |      |
               L1     L1
               |      |
               |__L2__|


    .. plot::
      :include-source:

      import pp

      c = pp.c.mzi(L0=0.1, L1=0, L2=10)
      pp.plotgds(c)

    """
    y = pp.call_if_func(y_model_factory)
    gc = siepic.ebeam_gc_te1550()
    wg_long = siepic.ebeam_wg_integral_1550(length=(2 * L0 + +L1 + L2) * 1e-6)
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


def mzi_simulation(**kwargs):
    """ Run a simulation on the netlist """
    circuit = mzi_circuit(**kwargs)

    simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
    result = simulation.simulate()

    f, s = result.data("input", "output")
    plt.plot(f, s)
    plt.title("MZI")
    plt.tight_layout()
    plt.show()


def mzi_variation(**kwargs):
    """ Run a montercarlo simulation on the netlist """
    circuit = mzi_circuit(**kwargs)

    simulation = MonteCarloSweepSimulation(circuit, 1500e-9, 1600e-9)
    runs = 10
    result = simulation.simulate(runs=runs)
    for i in range(1, runs + 1):
        f, s = result.data("input", "output", i)
        plt.plot(f, s)

    # The data located at the 0 position is the ideal values.
    f, s = result.data("input", "output", 0)
    plt.plot(f, s, "k")
    plt.title("MZI Monte Carlo")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mzi_simulation()
