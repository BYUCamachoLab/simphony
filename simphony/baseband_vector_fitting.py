import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import block_diag
import pandas as pd
from numpy.linalg import svd


class BVF_Options:
    def __init__(self, poles_estimation_threshold = 1e-1, 
                 model_error_threshold = 1e-3, 
                 max_iterations = 5, 
                 enforce_stability = True, 
                 alpha = 0.01,
                 beta = 1.5,
                 gamma = 0.95,
                 debug = True,
                 mode = "CVF"):
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

class Baseband_Model:
    def __init__(self, wvl_microns, center_wvl, S, order, options=None):
        if options == None:
            self.options = BVF_Options()
        else:
            self.options = options

        c = 299792458
        self.order = order
        self.wvl_microns = wvl_microns
        self.center_freq = c / (center_wvl * 1e-6)
        self.freqs = np.flip(c / (wvl_microns * 1e-6)) - self.center_freq
        self.sampling_freq = self.options.beta * (np.max(self.freqs) - np.min(self.freqs))
        self.poles = np.array([])
        self.digital_freq = 2 * np.pi * self.freqs / self.sampling_freq
        # self.digital_freq = 2 * np.pi * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order) / self.sampling_freq
        self.z = np.exp(1j*self.digital_freq)
        self.S = S
        self.error = float('inf')

    def initial_poles(self):
        digital_freq = 2 * np.pi * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order) / self.sampling_freq
        #digital_freq = 2 * np.pi * np.linspace(-self.sampling_freq / 2, self.sampling_freq / 2, self.order)
        return self.options.gamma * np.exp(1j*digital_freq)
    
    def compute_phi_matrices(self):
        phi1 = 1 / (self.z[0]-self.poles)
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
            D = np.diag(self.S)
            M = np.hstack((phi0, -D@phi1))
            V = self.S
            Q,R = np.linalg.qr(M,mode='reduced') 
            solutions = np.linalg.pinv(R)@Q.conj().T@V
            inital_residues = solutions[:self.order+1]
            self.weight_coefficients = solutions[self.order+1:]

            #weighting_term = 1 + self.weight_coefficients@phi1
            self.weights = self.calculate_weights(phi1)
            #print(np.max(np.abs(weighting_term-1)))

            if np.max(np.abs(self.weights-1)) < self.options.poles_estimation_threshold:
                self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
            # Calculate New Poles
            unity_column = np.ones((self.order, 1))
            A = np.diag(self.poles)
            #bw = unity_column
            weights_row = self.weight_coefficients.reshape((len(self.weight_coefficients), 1)).T
            self.poles, _ = np.linalg.eig(A-unity_column@weights_row)
            mask = np.abs(self.poles) > 1
            self.poles[mask] = 1 / (self.poles[mask])
            iter += 1

        print(f"Failed to reach poles estimation threshold after {iter} iterations")
        phi0, phi1 = self.compute_phi_matrices()
        self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
        self.error = self.compute_error()
        self.plot_model()
    
    def calculate_weights(self, phi1):
        return phi1@self.weight_coefficients
    
    def compute_response(self, wvl=None, freq=None):
        c = 299792458
        if freq is not None:
            digital_freq = 2 * np.pi * freq / self.sampling_freq
            z = np.exp(1j*digital_freq)
            response = np.full_like(z, self.residues[0], dtype=complex)
            for r, p in zip(self.residues[1:], self.poles):
                response += + r / (z - p)        
        elif wvl is not None:
            freq = np.flip(c / (wvl * 1e-6)) - self.center_freq
            digital_freq = 2 * np.pi * freq / self.sampling_freq
            z = np.exp(1j*digital_freq)
            response = np.full_like(z, self.residues[0], dtype=complex)
            for r, p in zip(self.residues[1:], self.poles):
                response += + r / (z - p)        
        else:
            freq = np.flip(c / (self.wvl_microns * 1e-6)) - self.center_freq
            digital_freq = 2 * np.pi * freq / self.sampling_freq
            z = np.exp(1j*digital_freq)
            response = np.full_like(z, self.residues[0], dtype=complex)
            for r, p in zip(self.residues[1:], self.poles):
                response += + r / (z - p)        

        # response1 = np.zeros((len(z)), dtype=complex)
        # for i in range(len(z)):
        #     response1[i] = self.residues[0] 
        #     for n in range(self.order):
        #         response1 += self.residues[n+1] / (z[i] - self.poles[n])

        
        return response
    
    def compute_error(self):
        return np.max(np.abs(self.S - self.compute_response()))

    def complex_ABCD_matrices(self):
        A = None
        B = None
        C = None
        D = None

        A = np.diag(self.poles)
        U, S, Vh = svd(np.array([[self.residues[1]]]))
        B = Vh
        for n in range(2, self.order + 1):
            print(n)
            U, S, Vh = svd([[self.residues[n]]])
            B = np.append(B, Vh)

        # B = B.reshape((1, B.shape[0]))
        B = B.reshape((B.shape[0], 1))

        C = self.residues[1:]
        D = np.array([self.residues[0]])

        return A, B, C, D

    def real_ABCD_matrices(self):
        A, B, C, D = self.complex_ABCD_matrices()

        A_hat = np.block([
            [np.real(A), -np.imag(A)], 
            [np.imag(A),  np.real(A)]
        ])
        
        B_hat = np.block([
            [np.real(B), np.zeros(B.shape)],
            [np.zeros(B.shape), np.real(B)]
        ])

        C_hat = np.block([
            [np.real(C), -np.imag(C)], 
            [np.imag(C),  np.real(C)]
        ])

        D_hat = np.block([
            [np.real(D), -np.imag(D)], 
            [np.imag(D),  np.real(D)]
        ])

        return A_hat, B_hat, C_hat, D_hat

    def plot_model(self):
        c = 299792458
        # wvl = np.linspace(1.5, 1.6, 100)
        # # wvl = np.flip(c / np.linspace(-self.sampling_freq / 2, self.sampling_freq / 2, 150))

        # freq = np.flip(c / (wvl * 1e-6)) - self.center_freq
        # response = self.compute_response(wvl=wvl)
        # plt.title("Magnitude")
        # plt.scatter(freq, np.abs(self.S)**2, label="Samples")
        # plt.plot(freq, np.abs(response)**2, label="Model")
        # plt.legend()
        # plt.show()
        num_measurements = len(self.freqs)
        freq = np.linspace(-self.sampling_freq / 2, self.sampling_freq / 2, int(self.options.beta * num_measurements))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.set_title("Magnitude")
        line_model, = ax1.plot(freq, np.abs(self.compute_response(freq=freq))**2, label=f"Model: order={self.order}, beta={self.options.beta}")
        line_samples, = ax1.plot(self.freqs, np.abs(self.S)**2, "r--", label="Samples")
        ax1.set_xlabel("Baseband Frequency")
        ax1.set_ylabel("Transmission")
        #ax1.legend()

        ax2.set_title("Phase")
        ax2.plot(freq, np.angle(self.compute_response(freq=freq)), label="Model")
        ax2.plot(self.freqs, np.angle(self.S), "r--", label="Samples")
        ax2.set_xlabel("Baseband Frequency")
        ax2.set_ylabel("Phase (radians)")
        #ax2.legend()

        fig.legend([line_model, line_samples], [line_model.get_label(), line_samples.get_label()], loc="upper center")
        plt.tight_layout()


        # plt.title("Magnitude")
        # plt.plot(freq, np.abs(self.compute_response(freq=freq))**2)
        # plt.plot(self.freqs, np.abs(self.S)**2, "r--")
        plt.show()
        pass
    
    def plot_poles(self):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        initial_poles = self.initial_poles()
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_thetagrids([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        ax.set_rlim(0.0, 1.0)
        ax.set_rlabel_position(0)
        ax.grid(True)
        #ax.grid(True)

        # Plot Initial Poles
        ax.scatter(np.angle(initial_poles), np.abs(initial_poles), label="Initial Poles")

        # Plot Current Poles
        ax.scatter(np.angle(self.poles), np.abs(self.poles), label="Optimal Poles")

        #ax.set_title("A line plot on a polar axis", va='bottom')
        ax.legend()
        plt.show()


