import pp
from simphony.library import siepic
from simphony.library.gdsfactory import sweep_simulation
from simphony.library.gdsfactory.mmi1x2 import mmi1x2
from simphony.netlist import Subcircuit


@pp.autoname
def mzi(L0=1, L1=100, L2=10, y_model_factory=mmi1x2):
    """ Mzi

    Args:
        L0: vertical length for both and top arms
        L1: bottom arm extra length, delta_length = 2*L1
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


    .. plot::
        :include-source:

        import simphony.library.gdsfactory as cl
        c = cl.mzi()

        simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
        result = simulation.simulate()

        f, s = result.data("input", "output")
        plt.plot(f, s)
        plt.title("MZI")
        plt.tight_layout()
        plt.show()

    """
    y = pp.call_if_func(y_model_factory)
    wg_long = siepic.ebeam_wg_integral_1550(length=(2 * L0 + 2 * L1 + L2) * 1e-6)
    wg_short = siepic.ebeam_wg_integral_1550(length=(2 * L0 + L2) * 1e-6)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("mzi")
    circuit.add(
        [
            (y, "splitter"),
            (y, "recombiner"),
            (wg_long, "wg_long"),
            (wg_short, "wg_short"),
        ]
    )

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("splitter", "E0", "wg_long", "n1"),
            ("splitter", "E1", "wg_short", "n1"),
            ("recombiner", "E0", "wg_long", "n2"),
            ("recombiner", "E1", "wg_short", "n2"),
        ]
    )
    circuit.elements["splitter"].pins["W0"] = "input"
    circuit.elements["recombiner"].pins["W0"] = "output"
    return circuit


if __name__ == "__main__":
    c = mzi()
    sweep_simulation(c)
