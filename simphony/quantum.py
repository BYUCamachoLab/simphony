"""Module for quantum simulation."""

from dataclasses import dataclass
from functools import partial
from typing import List, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike
from sax.saxtypes import Model
from sax.utils import get_ports
from scipy.stats import multivariate_normal

from simphony.exceptions import ShapeMismatchError
from simphony.simulation import SimDevice, Simulation, SimulationResult
from simphony.utils import dict_to_matrix, xpxp_to_xxpp, xxpp_to_xpxp


def plot_mode(means, cov, n=100, x_range=None, y_range=None, ax=None, **kwargs):
    """Plots the Wigner function of a single mode state.

    Parameters
    ----------
    means : ArrayLike
        The means of the X and P quadratures of the quantum state. For example,
        a coherent state :math:`\alpha = 3+4i` has means defined as
        :math:`\begin{bmatrix} 3 & 4 \\end{bmatrix}'. The shape of the means
        must be a length of 2.
    cov : ArrayLike
        The covariance matrix of the quantum state. For example, all coherent
        states has a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 &
        1/4 \\end{bmatrix}`. The shape of the matrix must be 2 x 2.
    n : int
        The number of points per axis to plot. Default is 100.
    x_range : tuple
        The range of the x axis to plot as a tuple, (eg. (-5,5)). Defualt
        attempts to find the range automatically.
    y_range : tuple
        The range of the y axis to plot as a tuple, (eg. (-5,5)). Defualt
        attempts to find the range automatically.
    ax : matplotlib.axes.Axes
        The axis to plot on, by default it creates a new figure.
    **kwargs :
        Keyword arguments to pass to matplotlib.pyplot.contourf.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if x_range is None:
        x_range = (
            means[0] - 5 * jnp.sqrt(cov[0, 0]),
            means[0] + 5 * jnp.sqrt(cov[0, 0]),
        )
    if y_range is None:
        y_range = (
            means[1] - 5 * jnp.sqrt(cov[1, 1]),
            means[1] + 5 * jnp.sqrt(cov[1, 1]),
        )
    x_max = jnp.max(jnp.abs(jnp.array(x_range)))
    y_max = jnp.max(jnp.abs(jnp.array(y_range)))
    r_max = jnp.max(jnp.array((x_max, y_max)))
    x_range = (-r_max, r_max)
    y_range = (-r_max, r_max)

    x = jnp.linspace(x_range[0], x_range[1], n)
    y = jnp.linspace(y_range[0], y_range[1], n)
    X, Y = jnp.meshgrid(x, y)
    pos = jnp.dstack((X, Y))
    dist = multivariate_normal(means, cov)
    pdf = dist.pdf(pos)
    ax.contourf(X, Y, pdf, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("P")
    return ax


class QuantumState(SimDevice):
    r"""Represents a quantum state in a quantum model as a covariance matrix.

    All quantum states are represented in the xpxp convention.

    Parameters
    ----------
    means : ArrayLike
        The means of the X and P quadratures of the quantum state. For example,
        a coherent state :math:`\alpha = 3+4i` has means defined as
        :math:`\begin{bmatrix} 3 & 4 \\end{bmatrix}'. The shape of the means
        must be 2 * N.
    cov : ArrayLike
        The covariance matrix of the quantum state. For example, all coherent
        states has a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 &
        1/4 \\end{bmatrix}`. The shape of the matrix must be 2 * N x 2 * N.
    ports : str or list of str
        The ports to which the quantum state is connected. Each mode
        corresponds in order to each port provided.
    convention : str
        The convention of the means and covariance matrix. Default is 'xpxp'.
    """

    def __init__(
        self,
        means: ArrayLike,
        cov: ArrayLike,
        ports: Union[str, List[str]] = None,
        convention: str = "xpxp",
    ) -> None:
        super().__init__(ports)
        if ports is None:
            self.N = int(len(means) / 2)
        else:
            self.N = len(ports)
            if means.shape != (2 * self.N,):
                raise ShapeMismatchError("The shape of the means must be 2 * N.")
            if cov.shape != (2 * self.N, 2 * self.N):
                raise ShapeMismatchError(
                    "The shape of the covariance matrix must \
                    be 2 * N x 2 * N."
                )
        self.means = means
        self.cov = cov
        self.convention = convention

    def to_xpxp(self) -> None:
        """Converts the means and covariance matrix to the xpxp convention."""
        if self.convention == "xxpp":
            self.means = xxpp_to_xpxp(self.means)
            self.cov = xxpp_to_xpxp(self.cov)
            self.convention = "xpxp"

    def to_xxpp(self) -> None:
        """Converts the means and covariance matrix to the xxpp convention."""
        if self.convention == "xpxp":
            self.means = xpxp_to_xxpp(self.means)
            self.cov = xpxp_to_xxpp(self.cov)
            self.convention = "xxpp"

    def modes(self, modes: Union[int, List[int]]):
        """Returns the mean and covariance matrix of the specified modes.

        Parameters
        ----------
        modes : int or list
            The modes to return.
        """
        if not hasattr(modes, "__iter__"):
            modes = [modes]
        if not all(mode < self.N for mode in modes):
            raise ValueError("Modes must be less than the number of modes.")
        modes = jnp.array(modes)
        inds = jnp.concatenate((modes, (modes + self.N)))
        if self.convention == "xpxp":
            means = xpxp_to_xxpp(self.means)
            cov = xpxp_to_xxpp(self.cov)
            means = means[inds]
            cov = cov[jnp.ix_(inds, inds)]
            means = xxpp_to_xpxp(means)
            cov = xxpp_to_xpxp(cov)
        else:
            means = self.means[inds]
            cov = self.cov[jnp.ix_(inds, inds)]
        return means, cov

    def _add_vacuums(self, n_vacuums: int):
        """Adds vacuum states to the quantum state.

        Parameters
        ----------
        n_vacuums : int
            The number of vacuum states to add.
        """
        N = self.N + n_vacuums
        means = jnp.concatenate((self.means, jnp.zeros(2 * n_vacuums)))
        cov = 0.25 * jnp.eye(2 * N)
        cov = cov.at[: 2 * self.N, : 2 * self.N].set(self.cov)
        self.means = means
        self.cov = cov
        self.N = N

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"\nConvention: {self.convention}\nMeans: {self.means}\nCov: \n{self.cov}"
        )

    def plot_mode(self, mode, n=100, x_range=None, y_range=None, ax=None, **kwargs):
        """Plots the Wigner function of the specified mode.

        Parameters
        ----------
        mode : int
            The mode to plot.
        n : int
            The number of points per axis to plot. Default is 100.
        x_range : tuple
            The range of the x axis to plot as a tuple, (eg. (-5,5)). Defualt
            attempts to find the range automatically.
        y_range : tuple
            The range of the y axis to plot as a tuple, (eg. (-5,5)). Defualt
            attempts to find the range automatically.
        ax : matplotlib.axes.Axes
            The axis to plot on, by default it creates a new figure.
        **kwargs
            Keyword arguments to pass to matplotlib.pyplot.contourf.
        """
        means, cov = self.modes(mode)

        return plot_mode(means, cov, n, x_range, y_range, ax, **kwargs)


def compose_qstate(*args: QuantumState) -> QuantumState:
    """Combines the quantum states of the input ports into a single quantum
    state.

    Parameters
    ----------
    args : QuantumState
        The quantum states to combine.
    """
    N = 0
    mean_list = []
    cov_list = []
    port_list = []
    for qstate in args:
        qstate.to_xpxp()
        N += qstate.N
        mean_list.append(qstate.means)
        cov_list.append(qstate.cov)
        port_list += qstate.ports

    means = jnp.concatenate(mean_list)
    covs = jnp.zeros((2 * N, 2 * N), dtype=float)
    left = 0

    for qstate in args:
        rowcol = qstate.N * 2 + left
        covs = covs.at[left:rowcol, left:rowcol].set(qstate.cov)
        left = rowcol
    return QuantumState(means, covs, port_list, convention="xpxp")


class CoherentState(QuantumState):
    """Represents a coherent state in a quantum model as a covariance matrix.

    Parameters
    ----------
    port : complex
        The port to which the coherent state is connected.
    alpha : str
        The complex amplitude of the coherent state.
    """

    def __init__(self, port: str, alpha: complex) -> None:
        self.alpha = alpha
        self.N = 1
        means = jnp.array([alpha.real, alpha.imag])
        cov = jnp.array([[1 / 4, 0], [0, 1 / 4]])
        ports = [port]
        super().__init__(means, cov, ports)


class SqueezedState(QuantumState):
    """Represents a squeezed state in a quantum model as a covariance matrix.

    Parameters
    ----------
    port : float
        The port to which the squeezed state is connected.
    r : str
        The squeezing parameter of the squeezed state.
    phi : float
        The squeezing phase of the squeezed state.
    alpha: complex, optional
        The complex displacement of the squeezed state. Default is 0.
    """

    def __init__(self, port: str, r: float, phi: float, alpha: complex = 0) -> None:
        self.r = r
        self.phi = phi
        self.N = 1
        means = jnp.array([alpha.real, alpha.imag])
        c, s = jnp.cos(phi / 2), jnp.sin(phi / 2)
        rot_mat = jnp.array([[c, -s], [s, c]])
        cov = (
            rot_mat
            @ ((1 / 4) * jnp.array([[jnp.exp(-2 * r), 0], [0, jnp.exp(2 * r)]]))
            @ rot_mat.T
        )
        ports = [port]
        super().__init__(means, cov, ports)


class TwoModeSqueezedState(QuantumState):
    """Represents a two mode squeezed state in a quantum model as a covariance
    matrix.

    This state is described by three parameters: a two-mode squeezing
    parameter r, and the two initial thermal occupations n_a and n_b.

    Parameters
    ----------
    r : float
        The two-mode squeezing parameter of the two mode squeezed state.
    n_a : float
        The initial thermal occupation of the first mode.
    n_b : float
        The initial thermal occupation of the second mode.
    port_a : str
        The port to which the first mode is connected.
    port_b : str
        The port to which the second mode is connected.
    """

    def __init__(
        self, r: float, n_a: float, n_b: float, port_a: str, port_b: str
    ) -> None:
        self.r = r
        self.n_a = n_a
        self.n_b = n_b
        self.N = 2
        means = jnp.array([0, 0, 0, 0])
        ca = (n_a + 1 / 2) * jnp.cosh(r) ** 2 + (n_b + 1 / 2) * jnp.sinh(r) ** 2
        cb = (n_b + 1 / 2) * jnp.cosh(r) ** 2 + (n_a + 1 / 2) * jnp.sinh(r) ** 2
        cab = (n_a + n_b + 1) * jnp.sinh(r) * jnp.cosh(r)
        cov = (
            jnp.array(
                [[ca, 0, cab, 0], [0, cb, 0, cab], [cab, 0, cb, 0], [0, cab, 0, ca]]
            )
            / 2
        )
        ports = [port_a, port_b]
        super().__init__(means, cov, ports)


class ThermalState(QuantumState):
    """Represents a thermal state in a quantum model as a covariance matrix.

    Parameters
    ----------
    port : str
        The port to which the thermal state is connected.
    nbar : float
        The thermal occupation or average photon number of the thermal state.
    """

    def __init__(self, port: str, nbar: float) -> None:
        self.nbar = nbar
        self.N = 1
        means = jnp.array([0, 0])
        cov = (2 * nbar + 1) / 4 * jnp.eye(2)
        ports = [port]
        super().__init__(means, cov, ports)


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

    def state(self, wl_ind: int = 0) -> QuantumState:
        """Returns the quantum state at a specific wavelength.

        Parameters
        ----------
        wl_ind : int, optional
            The wavelength index. Defaults to 0.
        """
        means = self.means[wl_ind]
        cov = self.cov[wl_ind]
        return QuantumState(means, cov, convention="xxpp")


def plot_quantum_result(
    result: QuantumResult,
    modes: list = None,
    wl_ind: int = 0,
    include_loss_modes=False,
):
    """Plot the means and covariance matrix of the quantum result.

    Parameters
    ----------
    result : QuantumResult
        The quantum simulation result.
    modes : list, optional
        The modes to plot. Defaults to all modes.
    wl_ind : int, optional
        The wavelength index to plot. Defaults to 0.
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
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axs = axs.flatten()
    # convert quantum result into quantum state
    means = result.means[wl_ind]
    cov = result.cov[wl_ind]
    for i, mode in enumerate(modes):
        mode = jnp.array([mode])
        inds = jnp.concatenate((mode, (mode + 1)))
        mu = means[inds]
        c = cov[jnp.ix_(inds, inds)]
        ax = axs[i]
        plot_mode(mu, c, x_range=(-6, 6), y_range=(-6, 6), ax=ax)
        ax.set_title(f"Mode {mode}")
    return axs


class QuantumSim(Simulation):
    """Quantum simulation.

    Parameters
    ----------
    ckt : sax.saxtypes.Model
        The circuit to simulate.
    wl : ArrayLike
        The array of wavelengths to simulate (in microns).
    **params
        Any other parameters to pass to the circuit.

    Examples
    --------
    >>> sim = QuantumSim(ckt=mzi, wl=wl, top={"length": 150.0}, bottom={"length": 50.0})
    """

    def __init__(self, ckt: Model, **kwargs) -> None:
        ckt = partial(ckt, **kwargs)
        if "wl" not in kwargs:
            raise ValueError("Must specify 'wl' (wavelengths to simulate).")
        super().__init__(ckt, kwargs["wl"])

    def add_qstate(self, qstate: QuantumState) -> None:
        """Add a quantum state to the simulation.

        Parameters
        ----------
        qstate : QuantumState
            The quantum state to add.
        """
        self.input = qstate

    @staticmethod
    def to_unitary(s_params):
        """This method converts s-parameters into a unitary transform by adding
        vacuum ports.

        The original ports maintain their index while new vacuum ports will
        always be the last n_ports.

        Parameters
        ----------
        s_params : ArrayLike
            s-parameters in the shape of (n_freq, n_ports, n_ports).

        Returns
        -------
        unitary : Array
            The unitary s-parameters of the shape (n_freq, 2*n_ports,
            2*n_ports).
        """
        n_freqs, n_ports, _ = s_params.shape
        new_n_ports = n_ports * 2
        unitary = jnp.zeros((n_freqs, new_n_ports, new_n_ports), dtype=complex)
        for f in range(n_freqs):
            unitary = unitary.at[f, :n_ports, :n_ports].set(s_params[f])
            unitary = unitary.at[f, n_ports:, n_ports:].set(s_params[f])
            for i in range(n_ports):
                val = jnp.sqrt(
                    1 - unitary[f, :n_ports, i].dot(unitary[f, :n_ports, i].conj())
                )
                unitary = unitary.at[f, n_ports + i, i].set(val)
                unitary = unitary.at[f, i, n_ports + i].set(-val)

        return unitary

    def run(self) -> QuantumResult:
        """Run the simulation."""
        ports = get_ports(self.ckt())
        n_ports = len(ports)
        # get the unitary s-parameters of the circuit
        s_params = dict_to_matrix(self.ckt())
        unitary = self.to_unitary(s_params)
        # get an array of the indices of the input ports
        input_indices = [ports.index(port) for port in self.input.ports]
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

            transform = transform.at[:n, :n].set(s_wl.real)
            transform = transform.at[:n, n:].set(-s_wl.imag)
            transform = transform.at[n:, :n].set(s_wl.imag)
            transform = transform.at[n:, n:].set(s_wl.real)

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
