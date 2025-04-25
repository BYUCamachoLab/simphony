from scipy.linalg import block_diag
import numpy as np
from numpy.linalg import svd
from scipy.signal import  StateSpace, dlsim, lsim
from simphony.utils import dict_to_matrix
from collections.abc import Iterable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class PoleResidueModel(ABC):
    def __init__(self) -> None:
        pass
    def plot_poles(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        initial_poles = self.initial_poles()
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        ax.set_rlim(0.0, 1.0)
        ax.set_rlabel_position(0)
        ax.grid(True)
        

        # Plot Initial Poles
        ax.scatter(np.angle(initial_poles), np.abs(initial_poles), label="Initial Poles")

        # Plot Current Poles
        ax.scatter(np.angle(self.poles), np.abs(self.poles), label="Optimal Poles")

        
        plt.show()
    
    def compute_error(self):
            return np.max(np.abs(self.S - self.compute_response())) 

class BVF_Options:
    def __init__(self, 
                 poles_estimation_threshold = 1e-1, 
                 model_error_threshold = 1e-3, 
                 max_iterations = 10, 
                 enforce_stability = True, 
                 alpha = 0.01,
                 beta = 20.0,
                 gamma = 0.95,
                 debug = True,
                 real_valued = True):
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.real_valued = real_valued

class IIRModelBaseband(PoleResidueModel):
    def __init__(self, wvl_microns, center_wvl, s_params, order, options=None):
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
        
        self.sampling_freq = self.options.beta * (self.freqs[-1] - self.freqs[0])

        self.poles = np.array([])
        self.residues = np.zeros((order, self.num_ports, self.num_ports), dtype=complex)

        self.digital_freq = 2 * np.pi * self.freqs / (self.sampling_freq)
        self.z = np.exp(1j*self.digital_freq)
        self.S = s_params
        self.error = float('inf')   

        self.A = None
        self.B = None
        self.C = None
        self.D = np.zeros((self.num_ports, self.num_ports), dtype=complex)
        self.time_response = None

        self.fit_model()

    def initial_poles(self):
        digital_freq = 2 * np.pi * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order)/self.sampling_freq
        return self.options.gamma * np.exp(1j*digital_freq)
    
    def compute_phi_matrices(self):
        phi1 = 1/(self.z[0]-self.poles)
        for z in self.z[1:]:
            phi1 = np.vstack((phi1, 1 / (z-self.poles)))
        
        unity_column = np.ones((len(self.z), 1))
        phi0 = np.hstack((unity_column, phi1))

        return phi0, phi1


    def fit_model(self):
        self.poles = self.initial_poles()

        iter = 1
        while iter < self.options.max_iterations:
            phi0, phi1 = self.compute_phi_matrices()
            M, V = self.compute_lstsq_matrices(phi0, phi1)
            Q, R = np.linalg.qr(M,mode='reduced') 
            solutions = np.linalg.pinv(R)@Q.conj().T@V

            # Calculate New Poles
            A = np.diag(self.poles)
            weight_coefficients = solutions[(self.num_ports**2)*(self.order+1):]
            weights_row = weight_coefficients.reshape((len(weight_coefficients), 1))
            unity_column = np.ones((self.order, 1))
            self.poles, _ = np.linalg.eig(A-unity_column@weights_row.T)
            mask = np.abs(self.poles) > 1
            self.poles[mask] = 1 / (self.poles[mask])

            if True:
                for i in range(self.num_ports):
                    for j in range(self.num_ports):
                        phi0, _ = self.compute_phi_matrices()
                        Q,R = np.linalg.qr(phi0,mode='reduced') 
                        solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        self.D[i, j] = solutions[0]
                        self.residues[:, i, j] = solutions[1:]

            
            iter += 1
    
    def compute_response(self, a, b, wvl=None):
        if wvl is None:
            wvl = self.wvl_microns

        c = 299792458
        freq = c / (wvl * 1e-6) - self.center_freq
        digital_freq = 2 * np.pi * freq / self.sampling_freq
        z = np.exp(1j*digital_freq)
        response = np.stack([self.D]*(z.shape[0]))
        for r, p in zip(self.residues[:, b, a], self.poles[:]):
            response[:, b, a] += r / (z - p)        

        return response[:, b, a]

    def plot(self, modes=None):
        if modes is None or modes == "all":
            n = self.num_ports
            fig, ax = plt.subplots(n, n, figsize=(10, 10))
            for i in range(n):
                for j in range(n):
                    ax[i, j].plot(np.abs(self.compute_response(i, j))**2)
                    ax[i, j].plot(np.abs(self.S[:, i, j])**2, "r--")

            plt.show()
        else:
            n = len(modes)
            for mode in modes:
                plt.plot(np.abs(self.compute_response(mode[0], mode[1]))**2)
                plt.plot(np.abs(self.S[:, mode[0], mode[1]])**2, "r--")
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
            sig = np.exp(1j*2*np.pi*t*0)
        
        sig = sig.reshape(-1, 1)
        impulse = np.hstack([np.real(sig), np.imag(sig)])

        t_out, yout, _ = dlsim(sys, impulse, t)
        yout = yout[:, 0] + 1j*yout[:, 1]
        self.time_response = (t_out, yout)

        return t_out, yout
        
    def compute_lstsq_matrices(self, phi0, phi1):
        D = []
        V = []
        for i in range(self.num_ports):
            for j in range(self.num_ports):
                D.append(np.diag(self.S[:, i, j]))
                V.append(self.S[:, i, j])

        phi_column = []
        for _D in D:
            phi_column.append(-_D@phi1)

        blocks = [phi0] * (self.num_ports*self.num_ports)
        M = block_diag(*blocks)

        phi_column = np.vstack(phi_column)
        M = np.hstack([M, phi_column])

        return M, np.hstack(V)
    
    def generate_sys_discrete(self):
        _A = []
        _B = []
        _C = []
        for p in self.poles:
            A_n = np.diag(np.full(self.num_ports, p))
            _A.append(A_n)
        
        for n in range(self.order):
            U, S, Vh = svd(self.residues[n, :, :])
            _B.append(Vh)
            _C.append(U@np.diag(S))

        A = block_diag(*_A)
        B = np.vstack(_B)
        C = np.hstack(_C)

        D = self.D
        
        return StateSpace(A, B, C, D, dt = np.abs(1/self.sampling_freq))
    
    def plot_time_response(self):
        if self.time_response is None:
            self.compute_time_response()

        t_out, yout = self.time_response

        plt.title("Time Response")
        plt.xlabel("Time")
        plt.ylabel("E-field Amplitude")
        plt.plot(t_out, np.abs(yout)**2)
        plt.axvline(x=0.59e-12, color='r', linestyle='--', linewidth=1, alpha=0.75)

    
    def plot_poles(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        initial_poles = self.initial_poles()
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        ax.set_rlim(0.0, 1.0)
        ax.set_rlabel_position(0)
        ax.grid(True)
        

        # Plot Initial Poles
        ax.scatter(np.angle(initial_poles), np.abs(initial_poles), label="Initial Poles")

        # Plot Current Poles
        ax.scatter(np.angle(self.poles), np.abs(self.poles), label="Optimal Poles")

        
        plt.show()
    
    

