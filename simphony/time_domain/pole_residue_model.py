from abc import ABC, abstractmethod
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from scipy.linalg import block_diag
from scipy.signal import StateSpace, dlsim, lsim

from simphony.utils import dict_to_matrix


class PoleResidueModel(ABC):
    def __init__(self) -> None:
        pass

    def plot_poles(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        initial_poles = self.initial_poles()
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        ax.set_rlim(0.0, 1.0)
        ax.set_rlabel_position(0)
        ax.grid(True)

        # Plot Initial Poles
        ax.scatter(
            np.angle(initial_poles), np.abs(initial_poles), label="Initial Poles"
        )

        # Plot Current Poles
        ax.scatter(np.angle(self.poles), np.abs(self.poles), label="Optimal Poles")

        plt.show()

    def compute_error(self):
        return np.max(np.abs(self.S - self.compute_response()))


class BVF_Options:
    def __init__(
        self,
        poles_estimation_threshold=1e-1,
        model_error_threshold=1e-3,
        max_iterations=10,
        enforce_stability=True,
        alpha=0.01,
        beta=20.0,
        gamma=0.95,
        debug=True,
        real_valued=True,
    ):
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.real_valued = real_valued
        self.mode = "Fast"


class IIRModelBaseband(PoleResidueModel):
    def __init__(self, wvl_microns, center_wvl, s_params, sampling_period, order, options=None):
        if options == None:
            self.options = BVF_Options()
        else:
            self.options = options

        c = 299792458
        self.order = order
        self.num_ports = s_params.shape[1]
        self.wvl_microns = wvl_microns
        self.center_freq = c / (center_wvl * 1e-6)

        self.freqs = c / (wvl_microns * 1e-6) - self.center_freq

        self.sampling_freq = 1/sampling_period
        self.options.beta = np.abs(self.sampling_freq / (self.freqs[-1] - self.freqs[0]))

        self.poles = np.array([])
        self.residues = np.zeros((order, self.num_ports, self.num_ports), dtype=complex)

        self.digital_freq = 2 * np.pi * self.freqs / (self.sampling_freq)
        self.z = np.exp(1j * self.digital_freq)
        self.S = s_params
        self.error = float("inf")

        self.A = None
        self.B = None
        self.C = None
        self.D = np.zeros((self.num_ports, self.num_ports), dtype=complex)
        self.time_response = None

        self.fit_model()

    def initial_poles(self):
        digital_freq = (
            2
            * np.pi
            * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order)
            / self.sampling_freq
        )
        return self.options.gamma * np.exp(1j * digital_freq)

    def compute_phi_matrices(self):
        phi1 = 1 / (self.z[0] - self.poles)
        for z in self.z[1:]:
            phi1 = np.vstack((phi1, 1 / (z - self.poles)))

        unity_column = np.ones((len(self.z), 1))
        phi0 = np.hstack((unity_column, phi1))

        return phi0, phi1

    def fit_model(self):
        self.poles = self.initial_poles()

        iter = 1
        while iter < self.options.max_iterations:
            phi0, phi1 = self.compute_phi_matrices()
            if self.options.mode == "Fast":
                M, B = self.compute_lstsq_matrices(phi0, phi1)
                weights, _, _, _ = np.linalg.lstsq(M, B)
                weights_row = weights.reshape((len(weights), 1))
                unity_column = np.ones((self.order, 1))

                A = np.diag(self.poles)
                self.poles, _ = np.linalg.eig(A - unity_column @ weights_row.T)
                mask = np.abs(self.poles) > 1
                self.poles[mask] = 1 / (self.poles[mask])

            else:
                M, V = self.compute_lstsq_matrices(phi0, phi1)
                solutions, _, _, _ = np.linalg.lstsq(M, V, rcond=None)

                # Calculate New Poles
                A = np.diag(self.poles)
                weight_coefficients = solutions[
                    (self.num_ports**2) * (self.order + 1) :
                ]
                weights_row = weight_coefficients.reshape((len(weight_coefficients), 1))
                unity_column = np.ones((self.order, 1))
                self.poles, _ = np.linalg.eig(A - unity_column @ weights_row.T)
                mask = np.abs(self.poles) > 1
                self.poles[mask] = 1 / (self.poles[mask])

            if True:
                for i in range(self.num_ports):
                    for j in range(self.num_ports):
                        phi0, _ = self.compute_phi_matrices()
                        # Q,R = np.linalg.qr(phi0,mode='reduced')
                        # solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        solutions, _, _, _ = np.linalg.lstsq(
                            phi0, self.S[:, i, j], rcond=None
                        )
                        self.D[i, j] = np.array(solutions[0])
                        self.residues[:, i, j] = solutions[1:]

            iter += 1

    def compute_response(self, a, b, wvl=None):
        if wvl is None:
            wvl = self.wvl_microns

        c = 299792458
        freq = c / (wvl * 1e-6) - self.center_freq
        digital_freq = 2 * np.pi * freq / self.sampling_freq
        z = np.exp(1j * digital_freq)
        response = np.stack([self.D] * (z.shape[0]))
        for r, p in zip(self.residues[:, b, a], self.poles[:]):
            response[:, b, a] += r / (z - p)

        return response[:, b, a]

    def plot(self, modes=None):
        if modes is None or modes == "all":
            n = self.num_ports
            fig, ax = plt.subplots(n, n, figsize=(10, 10))
            for i in range(n):
                for j in range(n):
                    ax[i, j].plot(np.abs(self.compute_response(i, j)) ** 2)
                    ax[i, j].plot(np.abs(self.S[:, i, j]) ** 2, "r--")

            plt.show()
        else:
            n = len(modes)
            for mode in modes:
                plt.plot(np.abs(self.compute_response(mode[0], mode[1])) ** 2)
                plt.plot(np.abs(self.S[:, mode[0], mode[1]]) ** 2, "r--")
                plt.show()

    def compute_error(self):
        return np.max(np.abs(self.S - self.compute_response()))

    def compute_time_response(self, sig=None, t=None):
        c = 299792458

        sys = self.generate_sys_discrete()

        if t is None:
            N = int(1000)
            T = 2e-11
            t = np.linspace(0, T, N)

        if sig is None:
            sig = np.exp(1j * 2 * np.pi * t * 0)

        sig = sig.reshape(-1, 1)
        impulse = np.hstack([np.real(sig), np.imag(sig)])

        t_out, yout, _ = dlsim(sys, impulse, t)
        yout = yout[:, 0] + 1j * yout[:, 1]
        self.time_response = (t_out, yout)

        return t_out, yout

    # def compute_lstsq_matrices(self, phi0, phi1):

    def compute_lstsq_matrices(self, phi0, phi1):
        if self.options.mode == "Fast":
            M = np.zeros(((self.num_ports**2) * self.order, self.order), dtype=complex)
            B = np.zeros(((self.num_ports**2) * self.order), dtype=complex)
            iter = 0
            for i in range(self.num_ports):
                for j in range(self.num_ports):
                    D = np.diag(self.S[:, i, j])
                    A1 = phi0
                    A2 = -D @ phi1
                    # Here we perform the modified gram schmidt orthonalization on the block matrix [A1, A2] described here:
                    # https://arxiv.org/pdf/2208.06194
                    # This allows us to implement the Fast Vector Fitting algorithm:
                    # https://scholar.googleusercontent.com/scholar?q=cache:u4aY-dn1tF8J:scholar.google.com/+piero+triverio+vector+fitting&hl=en&as_sdt=0,45

                    Q1, R11 = np.linalg.qr(A1)
                    R12 = Q1.conj().T @ A2
                    Q2, R22 = np.linalg.qr(A2 - Q1 @ R12)

                    # Q_total = np.block([Q1, Q2])
                    # R_total = np.block([[R11, R12], [np.zeros((50, 51)), R22]])

                    V = self.S[:, i, j]
                    M[(iter) * self.order : (iter + 1) * self.order, :] = R22
                    B[(iter) * self.order : (iter + 1) * self.order] = Q2.conj().T @ V
                    iter += 1
                    # plt.imshow(np.abs(M))
                    # plt.show()
                    # plt.plot(np.abs(B))
                    # plt.show()
                    pass

            return M, B
        else:
            D = []
            V = []
            for i in range(self.num_ports):
                for j in range(self.num_ports):
                    D.append(np.diag(self.S[:, i, j]))
                    V.append(self.S[:, i, j])

            phi_column = []
            for _D in D:
                phi_column.append(-_D @ phi1)

            blocks = [phi0] * (self.num_ports * self.num_ports)
            M = block_diag(*blocks)

            phi_column = np.vstack(phi_column)
            M = np.hstack([M, phi_column])

            return M, np.hstack(V)

    def generate_sys_discrete(self):
        A = np.zeros(
            (self.order * self.num_ports, self.order * self.num_ports), dtype=complex
        )
        B = np.zeros((self.order * self.num_ports, self.num_ports), dtype=complex)
        C = np.zeros((self.num_ports, self.order * self.num_ports), dtype=complex)
        for i in range(self.order):
            A[
                i * self.num_ports : (i + 1) * self.num_ports,
                i * self.num_ports : (i + 1) * self.num_ports,
            ] = self.poles[i] * np.eye(self.num_ports)
            B[i * self.num_ports : (i + 1) * self.num_ports, :] = np.eye(self.num_ports)
            C[:, i * self.num_ports : (i + 1) * self.num_ports] = self.residues[i, :, :]

        return StateSpace(A, B, C, self.D, dt=np.abs(1 / self.sampling_freq))

        # _A = []
        # _B = []
        # _C = []
        # for p in self.poles:
        #     A_n = np.diag(np.full(self.num_ports, p))
        #     _A.append(A_n)

        # for n in range(self.order):
        #     U, S, Vh = np.linalg.svd(self.residues[n, :, :])
        #     _B.append(Vh)
        #     _C.append(U@np.diag(S))

        # A = block_diag(*_A)
        # B = np.vstack(_B)
        # C = np.hstack(_C)

        # D = self.D

        # return StateSpace(A, B, C, self.D, dt = np.abs(1/self.sampling_freq))

    def plot_time_response(self):
        if self.time_response is None:
            self.compute_time_response()

        t_out, yout = self.time_response

        plt.title("Time Response")
        plt.xlabel("Time")
        plt.ylabel("E-field Amplitude")
        plt.plot(t_out, np.abs(yout) ** 2)
        plt.axvline(x=0.59e-12, color="r", linestyle="--", linewidth=1, alpha=0.75)

    def plot_poles(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        initial_poles = self.initial_poles()
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        ax.set_rlim(0.0, 1.0)
        ax.set_rlabel_position(0)
        ax.grid(True)

        # Plot Initial Poles
        ax.scatter(
            np.angle(initial_poles), np.abs(initial_poles), label="Initial Poles"
        )

        # Plot Current Poles
        ax.scatter(np.angle(self.poles), np.abs(self.poles), label="Optimal Poles")

        plt.show()
