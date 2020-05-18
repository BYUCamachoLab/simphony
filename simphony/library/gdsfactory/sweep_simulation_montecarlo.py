import matplotlib.pyplot as plt
import pp
from simphony.simulation import MonteCarloSweepSimulation


def sweep_simulation_montecarlo(circuit, wmin=1500e-9, wmax=1600e-9, runs=10, **kwargs):
    """ Run runs sweep simulations and plots variation
    """
    circuit = pp.call_if_func(circuit)

    simulation = MonteCarloSweepSimulation(circuit, wmin, wmax)
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
    from mzi import mzi

    sweep_simulation_montecarlo(mzi)
