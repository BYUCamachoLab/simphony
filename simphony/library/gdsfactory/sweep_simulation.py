import matplotlib.pyplot as plt
import pp
from simphony.simulation import SweepSimulation
from simphony.tools import freq2wl


def sweep_simulation(
    circuit, iport="input", oport="output", start=1500e-9, stop=1600e-9, **kwargs
):
    """ Run a simulation on the circuit
    """
    circuit = pp.call_if_func(circuit)

    simulation = SweepSimulation(circuit, start, stop)
    result = simulation.simulate()

    f, s = result.data(iport, oport)
    w = freq2wl(f) * 1e9
    plt.plot(w, s)
    plt.title(circuit.name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from add_gc import mzi

    c = mzi()
    print(c.name)
    sweep_simulation(mzi)
