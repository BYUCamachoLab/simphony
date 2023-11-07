"""Quantum states for quantum simulators."""

from typing import List, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from simphony.exceptions import ShapeMismatchError
from simphony.utils import xpxp_to_xxpp, xxpp_to_xpxp

from .simdevices import SimDevice


def plot_mode(means, cov, n=100, x_range=None, y_range=None, ax=None, **kwargs):
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
    """Represents a quantum state in a quantum model as a covariance matrix.
    All quantum states are represented in the xpxp convention. TODO: switch to
    xxpp convention.

    Parameters
    ----------
    means :
        The means of the X and P quadratures of the quantum state. For example,
        a coherent state :math:`\alpha = 3+4i` has means defined as
        :math:`\begin{bmatrix} 3 & 4 \\end{bmatrix}'. The shape of the means
        must be 2 * N.
    cov :
        The covariance matrix of the quantum state. For example, all coherent
        states has a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 &
        1/4 \\end{bmatrix}`. The shape of the matrix must be 2 * N x 2 * N.
    ports :
        The ports to which the quantum state is connected. Each mode
        corresponds in order to each port provided.
    convention :
        The convention of the means and covariance matrix. Default is 'xpxp'.
    """

    def __init__(
        self,
        means: jnp.ndarray,
        cov: jnp.ndarray,
        ports: Union[str, List[str]] = None,
        convention: str = "xpxp",
    ) -> None:
        super().__init__(ports)
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

    def modes(self, modes):
        """Returns the mean and covariance matrix of the specified modes.

        Parameters
        ----------
        modes :
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

    def _add_vacuums(self, n_vacuums):
        """Adds vacuum states to the quantum state.

        Parameters
        ----------
        n_vacuums :
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
        return super().__repr__() + f"\nMeans: {self.means}\nCov: \n{self.cov}"

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
        **kwargs :
            Keyword arguments to pass to matplotlib.pyplot.contourf.
        """
        means, cov = self.modes(mode)

        return plot_mode(means, cov, n, x_range, y_range, ax, **kwargs)


def compose_qstate(*args: QuantumState) -> QuantumState:
    """Combines the quantum states of the input ports into a single quantum
    state."""
    N = 0
    mean_list = []
    cov_list = []
    port_list = []
    ckts = []
    for qstate in args:
        # if not isinstance(qstate, QuantumState):
        #     raise TypeError("Input must be a QuantumState.")
        N += qstate.N
        mean_list.append(qstate.means)
        cov_list.append(qstate.cov)
        port_list += qstate.ports
    # TODO: Do we need to check if we have the same circuit?
    #     ckts.append(qstate.ckt)
    # # check if all ckts are the same
    # for ckt in ckts:
    #     if ckt is not ckts[0]:
    #         raise ValueError("All quantum states must be attached to the same circuit.")

    means = jnp.concatenate(mean_list)
    covs = jnp.zeros((2 * N, 2 * N), dtype=float)
    left = 0
    # TODO: Currently puts states into xpxp, but should change to xxpp
    for qstate in args:
        rowcol = qstate.N * 2 + left
        covs = covs.at[left:rowcol, left:rowcol].set(qstate.cov)
        left = rowcol
    return QuantumState(means, covs, port_list)


class CoherentState(QuantumState):
    """Represents a coherent state in a quantum model as a covariance matrix.

    Parameters
    ----------
    alpha :
        The complex amplitude of the coherent state.
    port :
        The port to which the coherent state is connected.
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
    r :
        The squeezing parameter of the squeezed state.
    phi :
        The squeezing phase of the squeezed state.
    port :
        The port to which the squeezed state is connected.
    alpha:
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


class TwoModeSqueezed(QuantumState):
    """Represents a two mode squeezed state in a quantum model as a covariance
    matrix. This state is described by three parameters: a two-mode squeezing
    parameter r, and the two initial thermal occupations n_a and n_b.

    Parameters
    ----------
    r :
        The two-mode squeezing parameter of the two mode squeezed state.
    n_a :
        The initial thermal occupation of the first mode.
    n_b :
        The initial thermal occupation of the second mode.
    port_a :
        The port to which the first mode is connected.
    port_b :
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
