"""Module for quantum simulation."""

from dataclasses import dataclass
from functools import partial
from math import comb
from typing import List, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike
from mpl_toolkits.mplot3d import Axes3D
from sax.saxtypes import Model
from sax.utils import get_ports

from simphony.exceptions import ShapeMismatchError
from simphony.simulation import SimDevice, Simulation, SimulationResult
from simphony.utils import (
    complex_multivariate_normal,
    dict_to_matrix,
    xpxp_to_xxpp,
    xxpp_to_xpxp,
)


def plot_mode(
    means, cov, weights, n=100, x_range=None, y_range=None, ax=None, **kwargs
):
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
    weights: ArrayLike
        A set of weights to do a weighted sum of states if means / cov contain
        more than one state each.
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
            jnp.min(means[:, 0] - 5 * jnp.sqrt(cov[:, 0, 0])),
            jnp.min(means[:, 0] + 5 * jnp.sqrt(cov[:, 0, 0])),
        )
    if y_range is None:
        y_range = (
            jnp.min(means[:, 1] - 5 * jnp.sqrt(cov[:, 1, 1])),
            jnp.min(means[:, 1] + 5 * jnp.sqrt(cov[:, 1, 1])),
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
    dist = [complex_multivariate_normal(means[i], cov[i]) for i in range(0, len(means))]

    pdf = jnp.zeros((n, n))
    for i in range(0, len(means)):
        pdf += weights[i] * dist[i].pdf(pos)

    pdf_min = jnp.min(pdf)
    pdf_max = jnp.max(pdf)
    pdf_range = jnp.max(jnp.abs(jnp.array([pdf_min, pdf_max])))

    if isinstance(ax, Axes3D):
        ax.plot_surface(X, Y, jnp.real(pdf), **kwargs)
    else:
        ax.contourf(
            X,
            Y,
            jnp.real(pdf),
            levels=jnp.linspace(-pdf_range, pdf_range, 12),
            **kwargs,
        )

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
        states have a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 &
        1/4 \\end{bmatrix}`. The shape of the matrix must be 2 * N x 2 * N.
    weights : ArrayLike
        When working with sums of gaussian states, the weights for each term.
        Used when means is a 2d array, where each term is a vector of means for
        a different state corresponding to each value in weights.  cov should also
        be upgraded to a 3d array.
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
        weights: ArrayLike = None,
        ports: Union[str, List[str]] = None,
        convention: str = "xpxp",
    ) -> None:
        super().__init__(ports)

        means = jnp.array(means).astype(jnp.complex64)
        cov = jnp.array(cov).astype(jnp.complex64)
        
        if len(means.shape) == 1:
            means = means[jnp.newaxis, :]
        if len(cov.shape) == 2:
            cov = cov[jnp.newaxis, :]

        self.states = means.shape[0]

        if weights is None:
            self.weights = jnp.ones(self.states) / self.states
        elif weights.shape != (self.states,):
            raise ShapeMismatchError(
                "The number of weights must be the number of means"
            )
        else:
            self.weights = weights

        #TODO: Should it have support if ports is a single string and not a list of strings?
        if ports is None:
            self.N = int(means.shape[-1] / 2)
        else:
            self.N = len(ports)

        if means.shape != (
            self.states,
            2 * self.N,
        ):
            raise ShapeMismatchError("The shape of the means must be 2 * N.")
        if cov.shape != (self.states, 2 * self.N, 2 * self.N):
            raise ShapeMismatchError(
                "The shape of the covariance matrix must \
                be 2 * N x 2 * N."
            )

        self.means = means
        self.cov = cov
        self.convention = convention

        self.interference_means = None
        self.interference_cov = None
        self.interference_weights = []
        self.compute_interference()

    def to_xpxp(self) -> None:
        """Converts the means and covariance matrix to the xpxp convention."""
        if self.convention == "xxpp":
            for i in range(self.states):
                self.means = self.means.at[i].set(xxpp_to_xpxp(self.means[i]))
                self.cov = self.cov.at[i].set(xxpp_to_xpxp(self.cov[i]))
            for i in range(0, 2 * comb(self.states, 2)):
                if self.interference_means != None:
                    self.interference_means = self.interference_means.at[i].set(
                        xxpp_to_xpxp(self.interference_means[i])
                    )
                    self.interference_cov = self.interference_cov.at[i].set(
                        xxpp_to_xpxp(self.interference_cov[i])
                    )
            self.convention = "xpxp"

    def to_xxpp(self) -> None:
        """Converts the means and covariance matrix to the xxpp convention."""
        if self.convention == "xpxp":
            for i in range(self.states):
                self.means = self.means.at[i].set(xpxp_to_xxpp(self.means[i]))
                self.cov = self.cov.at[i].set(xpxp_to_xxpp(self.cov[i]))
            for i in range(0, 2 * comb(self.states, 2)):
                if self.interference_means != None:
                    self.interference_means = self.interference_means.at[i].set(
                        xpxp_to_xxpp(self.interference_means[i])
                    )
                    self.interference_cov = self.interference_cov.at[i].set(
                        xpxp_to_xxpp(self.interference_cov[i])
                    )
            self.convention = "xxpp"

    def compute_interference(self):
        """Computer interference terms of the gaussian states that compose this
        QuantumState."""
        interference_means = []
        interference_cov = []
        interference_weights = []

        revert = False
        if self.convention == "xxpp":
            self.to_xpxp()
            revert = True

        state_list = range(0, self.states)
        for a in state_list:
            for b in state_list[a + 1 :]:
                exps = []
                rot = jnp.zeros((2 * self.N, 2 * self.N))
                form = jnp.zeros((2 * self.N, 2 * self.N))
                for i in range(self.N):
                    cov = self.cov[a]
                    pos = 2*i
                    cov = cov[pos : pos + 2, pos : pos + 2]
                    if jnp.allclose(cov, self.cov[b][pos : pos + 2, pos : pos + 2]):
                        eigenvalues, eigenvectors = jnp.linalg.eig(cov.real)
                        theta = jnp.arccos(eigenvectors[0][0].real)
                        r = -jnp.log(4 * eigenvalues[0]) / 2

                        exps.append(jnp.exp(-2 * r))
                        exps.append(jnp.exp(2 * r))

                        c = jnp.cos(-theta)
                        s = jnp.sin(-theta)
                        rot = (
                            rot.at[pos : pos + 2, pos : pos + 2]
                            .set(jnp.array([[c, s], [-s, c]]))
                            .real
                        )
                        form = (
                            form.at[pos : pos + 2, pos : pos + 2]
                            .set(
                                jnp.array([[0, jnp.exp(-2 * r)], [-jnp.exp(2 * r), 0]]).real
                            )
                        )
                    else:
                        raise ShapeMismatchError(
                            "Interference not implemented for different squeezing magnitude / angle"
                        )

                exps = jnp.array(exps).real
                real_mean = (self.means[a] + self.means[b]) / 2

                transform = rot.T @ form @ rot
                imag_mean = transform @ (self.means[a] - self.means[b]) / 2

                real_cov = 0.25 * jnp.diag(exps)
                real_cov = rot.T @ real_cov @ rot
                imag_cov = jnp.zeros((2 * self.N, 2 * self.N))

                interference_means.append(real_mean + 1j * imag_mean)
                interference_cov.append(real_cov + 1j * imag_cov)
                interference_weights.append(self.weights[a] * self.weights[b])

                interference_means.append(real_mean - 1j * imag_mean)
                interference_cov.append(real_cov - 1j * imag_cov)
                interference_weights.append(self.weights[a] * self.weights[b])

        self.interference_means = jnp.array(interference_means)
        self.interference_cov = jnp.array(interference_cov)
        self.interference_weights = jnp.array(interference_weights)

        if revert:
            self.to_xxpp()

    def modes(self, modes: Union[int, List[int]], include_interference=False):
        """Returns the mean and covariance matrix of the specified modes.

        Parameters
        ----------
        modes : int or list
            The modes to return.
        include_interference : Boolean
            Whether or not to return interference terms
        """
        if not hasattr(modes, "__iter__"):
            modes = [modes]

        modes = jnp.array(modes)
        if modes.ndim == 0:
            modes = modes[jnp.newaxis]
        if not all(mode < self.N for mode in modes):
            raise ValueError("Modes must be less than the number of modes.")
        inds = jnp.concatenate((modes, (modes + self.N)))

        # TODO switch convention instead of including 2 different calculations

        if self.convention == "xpxp":
            means = jnp.array([xpxp_to_xxpp(m)[inds] for m in self.means])
            cov = jnp.array([xpxp_to_xxpp(c)[jnp.ix_(inds, inds)] for c in self.cov])

            if include_interference and comb(self.states, 2) > 0:
                means = jnp.concatenate(
                    (
                        means,
                        jnp.array(
                            [xpxp_to_xxpp(m)[inds] for m in self.interference_means]
                        ),
                    )
                )
                cov = jnp.concatenate(
                    (
                        cov,
                        jnp.array(
                            [
                                xpxp_to_xxpp(c)[jnp.ix_(inds, inds)]
                                for c in self.interference_cov
                            ]
                        ),
                    )
                )

            means = jnp.array([xxpp_to_xpxp(m) for m in means])
            cov = jnp.array([xxpp_to_xpxp(c) for c in cov])
        else:
            means = jnp.array([m[inds] for m in self.means])
            cov = jnp.array([c[jnp.ix_(inds, inds)] for c in self.cov])

            if include_interference and comb(self.states, 2) > 0:
                means = jnp.concatenate(
                    (means, jnp.array([m[inds] for m in self.interference_means]))
                )
                cov = jnp.concatenate(
                    (
                        cov,
                        jnp.array(
                            [c[jnp.ix_(inds, inds)] for c in self.interference_cov]
                        ),
                    )
                )

        if include_interference and comb(self.states, 2) > 0:
            weights = jnp.concatenate(
                (jnp.square(self.weights), self.interference_weights)
            )
        else:
            weights = self.weights

        return weights, means, cov

    def _add_vacuums(self, n_vacuums: int):
        """Adds vacuum states to the quantum state.

        Parameters
        ----------
        n_vacuums : int
            The number of vacuum states to add.
        """
        revert = False
        if self.convention == "xxpp":
            revert = True
            self.to_xpxp()

        N = self.N + n_vacuums
        means = jnp.concatenate(
            (self.means, jnp.zeros((self.states, 2 * n_vacuums))), axis=1
        )
        cov = jnp.array([0.25 * jnp.eye(2 * N)] * self.states, dtype=jnp.complex64)
        for i in range(self.states):
            cov = cov.at[i, : 2 * self.N, : 2 * self.N].set(self.cov[i])
        self.means = means
        self.cov = cov
        self.N = N

        if revert:
            self.to_xxpp()

        self.compute_interference()

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"\nConvention: {self.convention}\nMeans: {self.means}\nCov: \n{self.cov}\nWeights: \n{self.weights}"
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
        weights, means, cov = self.modes(mode, include_interference=True)

        return plot_mode(means, cov, weights, n, x_range, y_range, ax, **kwargs)

    def plot_quantum_state(self, modes: list = None):
        """Plot the means and covariance matrix of the quantum result.

        Parameters
        ----------
        result : QuantumResult
            The quantum simulation result.
        modes : list, optional
            The modes to plot. Defaults to all modes.
        """

        # calculate the number of modes to plot
        if modes is None:
            n_modes = self.N
            modes = jnp.linspace(0, int(self.N) - 1, int(self.N), dtype=int)
        n_modes = len(modes)

        # create a grid of plots, a single plot for each mode
        # make subplots into a square grid
        n_rows = int(n_modes**0.5)
        n_cols = int(n_modes**0.5)
        while n_rows * n_cols < n_modes:
            if n_rows < n_cols:
                n_rows += 1
            else:
                n_cols += 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

        # TODO: Better computation of x_range / y_range (currently the function parameters are unused)

        # graph all the modes
        if not hasattr(axs, "__iter__"):
            self.plot_mode(0, x_range=(-6, 6), y_range=(-6, 6), ax=axs)
            axs.set_title(f"Mode 0")
        else:
            axs = axs.flatten()
            for i, mode in enumerate(modes):
                ax = axs[i]
                self.plot_mode(mode, x_range=(-6, 6), y_range=(-6, 6), ax=ax)
                ax.set_title(f"Mode {mode}")

        return axs

    def homodyne_measurement(self, mode, theta=0, n=100, x_range=None, y_range=None):
        """Performs a homodyne measurement along a certain direction given by
        theta. Does NOT modify the QuantumState object.

        Parameters
        ----------
        mode : integer
            The mode on which to perform homodyne detection.
        theta : integer
            The direction in phase space to perform homodyne detection. 0 is X, pi/2 is P
        n : integer
            The number of points on which to evaluate the wigner function before projection.
        x_range : List
            The range of the detection along the axis determined by theta
        y_range : List
            The range in the direction perpindicular to theta before projection.
        """

        if mode >= self.N:
            raise ShapeMismatchError("The mode must be less than the number of modes")

        weights, means, cov = self.modes(mode, include_interference=True)

        c, s = jnp.cos(-theta), jnp.sin(-theta)
        R = jnp.array([[c, -s], [s, c]])

        means = (R @ means.T).T
        cov = R @ cov @ R.T

        if x_range == None:
            x_max = 0
            # Only includes non-interference states
            for i in range(self.states):
                max = jnp.abs(means[i][0]) + 6 * jnp.abs(cov[i][0][0])
                if max > x_max:
                    x_max = max

            x_range = (-x_max, x_max)

        if y_range == None:
            y_max = 0
            # Only includes non-interference states
            for i in range(self.states):
                max = jnp.abs(means[i][1]) + 6 * jnp.abs(cov[i][1][1])
                if max > y_max:
                    y_max = max
            y_range = (-y_max, y_max)

        x = jnp.linspace(x_range[0], x_range[1], n)
        y = jnp.linspace(y_range[0], y_range[1], n)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((X, Y))
        dist = [
            complex_multivariate_normal(means[i], cov[i]) for i in range(0, len(means))
        ]

        wigner = jnp.zeros((n, n), dtype=complex)
        for i in range(0, len(means)):
            wigner += weights[i] * dist[i].pdf(pos)

        homodyne = jnp.zeros(n)
        for height_pdf in wigner:
            homodyne += height_pdf.real

        return x, homodyne


def compose_qstate(*args: QuantumState) -> QuantumState:
    """Combines the quantum states of the input ports into a single quantum
    state.

    Parameters
    ----------
    args : QuantumState
        The quantum states to combine.
    """

    N = 0
    states = []
    mean_list = []
    cov_list = []
    weight_list = []
    port_list = []
    for qstate in args:
        qstate.to_xpxp()
        N += qstate.N
        states.append(qstate.states)
        port_list += qstate.ports

    # All possible combinations of states (density matrix terms)
    new_states = jnp.indices(states, dtype=int).reshape(len(states), -1).T

    for state in new_states:
        mean_list.append(
            jnp.concatenate([qstate.means[state[i]] for i, qstate in enumerate(args)])
        )

        cov = jnp.zeros((2 * N, 2 * N), dtype=complex)
        left = 0
        for i, qstate in enumerate(args):
            rowcol = qstate.N * 2 + left
            cov = cov.at[left:rowcol, left:rowcol].set(qstate.cov[state[i]])
            left = rowcol

        cov_list.append(cov)

        weight = 1.0
        for i, qstate in enumerate(args):
            weight *= qstate.weights[state[i]]
        weight_list.append(weight)

    return QuantumState(
        jnp.array(mean_list, dtype=complex),
        jnp.array(cov_list),
        jnp.array(weight_list),
        port_list,
        convention="xpxp",
    )


def apply_unitary(
    unitary: ArrayLike, qstate: QuantumState, modes: Union[int, List[int]]
) -> QuantumState:
    if not hasattr(modes, "__iter__"):
        modes = [modes]
    if not all(mode < qstate.N for mode in modes):
        raise ValueError("Modes must be less than the number of modes.")

    modes = jnp.array(modes)
    inds = jnp.concatenate((modes, (modes + qstate.N)))

    revert = False
    if qstate.convention == "xpxp":
        qstate.to_xxpp()
        revert = True

    weights, input_means, input_cov = qstate.modes(modes)

    n = unitary.shape[0]
    transform = jnp.zeros((n * 2, n * 2))

    transform = transform.at[:n, :n].set(unitary.real)
    transform = transform.at[:n, n:].set(-unitary.imag)
    transform = transform.at[n:, :n].set(unitary.imag)
    transform = transform.at[n:, n:].set(unitary.real)

    output_means = []
    output_cov = []

    for i in range(qstate.states):
        output_means.append(transform @ input_means[i].T)
        output_cov.append(transform @ input_cov[i] @ transform.T)

    # TODO: Possibly implement tolerance for small numbers
    # convert small numbers to zero
    # output_means[abs(output_means) < 1e-10] = 0
    # output_cov[abs(output_cov) < 1e-10] = 0

    result = QuantumState(output_means, output_cov, weights=weights, convention="xxpp")

    if revert:
        result.to_xpxp()

    return result


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
        super().__init__(means=means, cov=cov, ports=ports)


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
        super().__init__(means=means, cov=cov, ports=ports)


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
        super().__init__(means=means, cov=cov, ports=ports)


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
        super().__init__(means=means, cov=cov, ports=ports)


class CatState(QuantumState):
    """Represents a cat state in a quantum model as 4 means and covariance
    matrices.

    Parameters
    ----------
    port : str
        The port to which the cat state is connected.
    alpha : complex
        Alpha for the coherent states that compose the cat state.
    """

    def __init__(self, port: str, alpha: complex) -> None:
        self.N = 1
        means = jnp.array([[alpha.real, alpha.imag], [-alpha.real, -alpha.imag]])
        cov = jnp.array([[[0.25, 0], [0, 0.25]]] * 2)
        weights = jnp.array([0.7071, 0.7071])
        ports = [port]
        super().__init__(means=means, cov=cov, weights=weights, ports=ports)


# TODO Add NumberState, GKPState


@dataclass
class QuantumResult(SimulationResult):
    """Quantum simulation results."""

    s_params: jnp.ndarray
    input_state: QuantumState
    output_states: List[QuantumState]
    wl: jnp.ndarray
    n_ports: int

    def state(self, wl_ind: int = 0) -> QuantumState:
        """Returns the quantum state at a specific wavelength.

        Parameters
        ----------
        wl_ind : int, optional
            The wavelength index. Defaults to 0.
        """
        return self.output_states[wl_ind]


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

    return result.state(wl_ind).plot_quantum_state(modes)


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

        # TODO: Check to see if two states are added to the same port. This causes simphony to crash right now
        #raise ShapeMismatchError("Cannot attach two quantum states to the same port")

        input_indices = [ports.index(port) for port in self.input.ports]

        # create vacuum ports for each extra mode in the unitary matrix
        n_modes = unitary.shape[1]
        n_vacuum = n_modes - len(input_indices)
        self.input._add_vacuums(n_vacuum)

        input_indices += [i for i in range(n_modes) if i not in input_indices]
        output_states = []
        for wl_ind in range(len(self.wl)):
            s_wl = unitary[wl_ind]
            output_state = apply_unitary(s_wl, self.input, input_indices)
            output_states.append(output_state)

        return QuantumResult(
            s_params=s_params,
            input_state=self.input,
            output_states=output_states,
            n_ports=n_ports,
            wl=self.wl,
        )
