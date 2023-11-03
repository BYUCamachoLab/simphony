"""Module for quantum simulation."""

from dataclasses import dataclass

import jax.numpy as jnp
import matplotlib.pyplot as plt

from simphony.circuit import Circuit

from .quantum_states import QuantumState
from .simulation import Simulation, SimulationResult


@dataclass
class QuantumResult(SimulationResult):
    """Quantum simulation results."""

    s_params: jnp.ndarray
    input_means: jnp.ndarray
    input_cov: jnp.ndarray
    transforms: jnp.ndarray
    means: jnp.ndarray
    cov: jnp.ndarray
    wl: jnp.ndarray
    n_ports: int


def plot_quantum_result(
    result: QuantumResult,
    modes: list = None,
    freq_ind: int = 0,
    include_loss_modes=False,
):
    """Plot the means and covariance matrix of the quantum result.

    Parameters
    ----------
    result : QuantumResult
        The quantum simulation result.
    modes : list, optional
        The modes to plot. Defaults to all modes.
    freq_ind : int, optional
        The frequency index to plot. Defaults to 0.
    include_loss_modes : bool, optional
        Whether to include the loss modes in the plot. Defaults to False.
    """
    # create a grid of plots, a single plot for each mode
    if modes is None:
        n_modes = result.n_ports * 2 if include_loss_modes else result.n_ports
        modes = jnp.linspace(0, int(n_modes) - 1, int(n_modes), dtype=int)
    n_modes = len(modes)
    # make subplots into a square grid
    n_rows = int(n_modes**0.5)
    n_cols = int(n_modes**0.5)
    if n_rows * n_cols < n_modes:
        n_rows += 1
        n_cols += 1
    print(n_rows, n_cols, n_modes)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.flatten()
    # convert quantum result into quantum state
    means = result.means[freq_ind]
    cov = result.cov[freq_ind]
    qstate = QuantumState(means, cov, result.n_ports * 2, "xxpp")
    for i, mode in enumerate(modes):
        # plot the means
        ax = axs[i]
        qstate.plot_mode(mode, x_range=(-6, 6), y_range=(-6, 6), ax=ax)
    return axs


class QuantumSim(Simulation):
    """Quantum simulation."""

    def __init__(self, ckt: Circuit, wl: jnp.ndarray, input: QuantumState) -> None:
        super().__init__(ckt, wl)

        # ensure that all ports of the input are connected to unique ports on the circuit
        if len(input.ports) != len(set(input.ports)):
            raise ValueError("Input ports are not unique")
        for port in input.ports:
            if port not in self.ckt._oports:
                raise ValueError("Input port not found in circuit")
        self.input = input

    @staticmethod
    def to_unitary(s_params):
        """This method converts s-parameters into a unitary transform by adding
        vacuum ports. The original ports maintain their index while new vacuum
        ports will always be the last n_ports.

        Parameters
        ----------
        s_params : jnp.ndarray
            s-parameters in the shape of (n_freq, n_ports, n_ports).

        Returns
        -------
        unitary : jnp.ndarray
            The unitary s-parameters of the shape (n_freq, 2*n_ports,
            2*n_ports).
        """
        n_freqs, n_ports, _ = s_params.shape
        new_n_ports = n_ports * 2
        unitary = jnp.zeros((n_freqs, new_n_ports, new_n_ports), dtype=complex)
        for f in range(n_freqs):
            unitary[f, :n_ports, :n_ports] = s_params[f]
            unitary[f, n_ports:, n_ports:] = s_params[f]
            # print(unitary[f, :n_ports, 0])
            for i in range(n_ports):
                val = jnp.sqrt(
                    1 - unitary[f, :n_ports, i].dot(unitary[f, :n_ports, i].conj())
                )
                # print(val, i)
                unitary[f, n_ports + i, i] = val
                unitary[f, i, n_ports + i] = -val

        return unitary

    def run(self) -> QuantumResult:
        """Run the simulation."""
        n_ports = len(self.ckt._oports)
        # get the unitary s-parameters of the circuit
        s_params = self.ckt.s_params(self.wl)
        unitary = self.to_unitary(s_params)
        # get an array of the indices of the input ports
        input_indices = [self.ckt._oports.index(port) for port in self.input.ports]
        # create vacuum ports for each extra mode in the unitary matrix
        n_modes = unitary.shape[1]
        n_vacuum = n_modes - len(input_indices)
        self.input._add_vacuums(n_vacuum)
        input_indices += [i for i in range(n_modes) if i not in input_indices]
        self.input.to_xxpp()
        input_means, input_cov = self.input.modes(input_indices)

        transforms = []
        means = []
        covs = []
        for wl_ind in range(len(self.wl)):
            s_wl = unitary[wl_ind]
            transform = jnp.zeros((n_modes * 2, n_modes * 2))
            n = n_modes

            transform[:n, :n] = s_wl.real
            transform[:n, n:] = -s_wl.imag
            transform[n:, :n] = s_wl.imag
            transform[n:, n:] = s_wl.real

            output_means = transform @ input_means.T
            output_cov = transform @ input_cov @ transform.T

            # TODO: Possibly implement tolerance for small numbers
            # convert small numbers to zero
            # output_means[abs(output_means) < 1e-10] = 0
            # output_cov[abs(output_cov) < 1e-10] = 0

            transforms.append(transform)
            means.append(output_means)
            covs.append(output_cov)

        return QuantumResult(
            s_params=s_params,
            input_means=input_means,
            input_cov=input_cov,
            transforms=jnp.stack(transforms),
            means=jnp.stack(means),
            cov=jnp.stack(covs),
            n_ports=n_ports,
            wl=self.wl,
        )
