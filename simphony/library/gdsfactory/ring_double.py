import pp
from simphony.library import siepic
from simphony.library.gdsfactory import sweep_simulation
from simphony.netlist import Subcircuit


@pp.autoname
def ring_double(
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

    W0 = 'n1'
    E0 = 'n2'

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
        [(coupler, "ct"), (coupler, "cb"), (waveguide, "wl"), (waveguide, "wr"),]
    )

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("cb", "N0", "wl", "n1"),
            ("wl", "n2", "ct", "N1"),
            ("ct", "N0", "wr", "n2"),
            ("wr", "n1", "cb", "N1"),
        ]
    )
    circuit.elements["cb"].pins["W0"] = "input"
    circuit.elements["cb"].pins["E0"] = "output"
    circuit.elements["ct"].pins["W0"] = "drop"
    circuit.elements["ct"].pins["E0"] = "cdrop"
    return circuit


if __name__ == "__main__":
    c = ring_double()
    sweep_simulation(c)
