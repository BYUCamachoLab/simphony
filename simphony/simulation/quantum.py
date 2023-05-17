"""
Module for quantum simulation.
"""

from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False

from .simulation import Simulation, SimulationResult
from .simdevices import SimDevice
from simphony.circuit import Circuit
from simphony.exceptions import ShapeMismatchError
from simphony.utils import xxpp_to_xpxp, xpxp_to_xxpp


class QuantumState(SimDevice):
    """
    Represents a quantum state in a quantum model as a covariance matrix. All
    quantum states are represented in the xpxp convention. TODO: switch to xxpp
    convention

    Parameters
    ----------
    N :
        The number of modes in the quantum state.
    means :
        The means of the X and P quadratures of the quantum state. For example,
        a coherent state :math:`\alpha = 3+4i` has means defined as
        :math:`\begin{bmatrix} 3 & 4 \end{bmatrix}'. The shape of the means
        must be 2 * N.
    cov :
        The covariance matrix of the quantum state. For example, all coherent
        states has a covariance matrix of :math:`\begin{bmatrix} 1/4 & 0 \\ 0 &
        1/4 \end{bmatrix}`. The shape of the matrix must be 2 * N x 2 * N.
    ports :
        The ports to which the quantum state is connected. Each mode
        corresponds in order to each port provided.
    convention :
        The convention of the means and covariance matrix. Default is 'xpxp'.
    """

    def __init__(
        self, ckt, ports, means: jnp.ndarray, cov: jnp.ndarray, convention="xpxp"
    ) -> None:
        super().__init__(ckt, ports)
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
        self.ports = ports
        self.convention = convention

    def to_xpxp(self) -> None:
        """
        Converts the means and covariance matrix to the xpxp convention.
        """
        if self.convention == "xxpp":
            self.means = xxpp_to_xpxp(self.means)
            self.cov = xxpp_to_xpxp(self.cov)
            self.convention = "xpxp"

    def to_xxpp(self) -> None:
        """
        Converts the means and covariance matrix to the xxpp convention.
        """
        if self.convention == "xpxp":
            self.means = xpxp_to_xxpp(self.means)
            self.cov = xpxp_to_xxpp(self.cov)
            self.convention = "xxpp"

    def modes(self, modes):
        """
        Returns the mean and covariance matrix of the specified modes.
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
        """
        Adds vacuum states to the quantum state.
        Parameters
        ----------
        n_vacuums :
            The number of vacuum states to add.
        """
        N = self.N + n_vacuums
        means = jnp.concatenate((self.means, jnp.zeros(2 * n_vacuums)))
        cov = 0.25 * jnp.eye(2 * N)
        cov[: 2 * self.N, : 2 * self.N] = self.cov
        self.means = means
        self.cov = cov
        self.N = N
    
    def __repr__(self) -> str:
        return super().__repr__() + f"\nMeans: {self.means}\nCov: \n{self.cov}"


def compose_qstate(*args: QuantumState) -> QuantumState:
    """
    Combines the quantum states of the input ports into a single quantum state.
    """
    N = 0
    mean_list = []
    cov_list = []
    port_list = []
    ckts = []
    for qstate in args:
        # if not isinstance(qstate, QuantumState):
        #     raise TypeError("Input must be a QuantumState.")
        ckts.append(qstate.ckt)
        N += qstate.N
        mean_list.append(qstate.means)
        cov_list.append(qstate.cov)
        port_list += qstate.ports
    # check if all ckts are the same
    for ckt in ckts:
        if ckt is not ckts[0]:
            raise ValueError("All quantum states must be attached to the same circuit.")
    means = jnp.concatenate(mean_list)
    covs = jnp.zeros((2 * N, 2 * N), dtype=float)
    left = 0
    # TODO: Currently puts states into xpxp, but should change to xxpp
    for qstate in args:
        rowcol = qstate.N * 2 + left
        covs[left:rowcol, left:rowcol] = qstate.cov
        left = rowcol
    return QuantumState(ckts[0], port_list, means, covs)


class CoherentState(QuantumState):
    """
    Represents a coherent state in a quantum model as a covariance matrix.
    Parameters
    ----------
    alpha :
        The complex amplitude of the coherent state.
    port :
        The port to which the coherent state is connected.
    """

    def __init__(self, ckt, port, alpha: complex) -> None:
        self.alpha = alpha
        self.N = 1
        means = jnp.array([alpha.real, alpha.imag])
        cov = jnp.array([[1 / 4, 0], [0, 1 / 4]])
        ports = [port]
        super().__init__(ckt, ports, means, cov)


class SqueezedState(QuantumState):
    """
    Represents a squeezed state in a quantum model as a covariance matrix.
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

    def __init__(self, ckt, port, r: float, phi: float, alpha: complex = 0) -> None:
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
        super().__init__(ckt, ports, means, cov)


class TwoModeSqueezed(QuantumState):
    """
    Represents a two mode squeezed state in a quantum model as a covariance matrix.
    This state is described by three parameters: a two-mode squeezing parameter r,
    and the two initial thermal occupations n_a and n_b.
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
        self, ckt, r: float, n_a: float, n_b: float, port_a, port_b
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
        super().__init__(ckt, ports, means, cov)


@dataclass
class QuantumResult(SimulationResult):
    """
    Quantum simulation results
    """

    s_params: jnp.ndarray
    input_means: jnp.ndarray
    input_cov: jnp.ndarray
    transforms: jnp.ndarray
    means: jnp.ndarray
    cov: jnp.ndarray
    wl: jnp.ndarray
    n_ports: int


class QuantumSim(Simulation):
    """
    Quantum simulation
    """

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
        """
        This method converts s-parameters into a unitary transform by adding
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

    def run(self):
        """
        Run the simulation
        """
        n_ports = len(self.ckt._oports)
        # get the unitary s-parameters of the circuit
        s_params = self.ckt.s_params(self.wl)
        unitary = self.to_unitary(s_params)
        # get an array of the indices of the input ports
        input_indices = [self.ckt._oports.index(port) for port in self.input.ports]
        # create vacuum ports for each every extra mode in the unitary matrix
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
            n_ports = n_ports,
            wl = self.wl,
        )