import pp
from simphony.library import siepic
from simphony.library.gdsfactory import sweep_simulation
from simphony.netlist import Subcircuit


@pp.autoname
def ring_double_siepic(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    bend_radius=5,
    length_y=2,
    coupler=siepic.ebeam_dc_halfring_straight,
    waveguide=siepic.ebeam_wg_integral_1550,
):
    """ double bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical waveguides (wyl: left, wyr: right)

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring(wg_width=0.5, gap=0.2, length_x=4, bend_radius=5, length_y=2)
      pp.plotgds(c)


    .. plot::
        :include-source:

        import simphony.library.gdsfactory as cl
        c = cl.ring()
        cl.sweep_simulation(c)
    """

    waveguide = pp.call_if_func(waveguide)
    coupler = pp.call_if_func(coupler)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("mzi")
    circuit.add(
        [(coupler, "ct"), (coupler, "cb"), (waveguide, "wl"), (waveguide, "wr")]
    )

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("cb", "2", "wl", "n1"),
            ("wl", "n2", "ct", "n4"),
            ("ct", "2", "wr", "n2"),
            ("wr", "n1", "cb", "n4"),
        ]
    )
    circuit.elements["cb"].pins["n4"] = "input"
    circuit.elements["cb"].pins["n3"] = "output"
    circuit.elements["ct"].pins["n3"] = "drop"
    circuit.elements["ct"].pins["n4"] = "cdrop"
    return circuit


if __name__ == "__main__":
    c = ring_double_siepic()
    sweep_simulation(c)
