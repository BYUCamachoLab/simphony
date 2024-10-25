import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import block_diag
import pandas as pd
from numpy.linalg import svd
from scipy.signal import  StateSpace, dlsim, lsim
from simphony.utils import dict_to_matrix
from collections.abc import Iterable
from scipy.signal import convolve, find_peaks, peak_widths
from scipy.optimize import curve_fit
from functools import partial
import warnings

def cross_covariance(X, Y):
    (X - X.mean()) * (Y - Y.mean()).conj().T

def covariance_X_plus_Y(cov_X, cov_Y):
    pass



class State_Space_Model:
    def __init__(self):
        self.continuous = None
        self.discrete = None
    
    def step(self, u, t=None, x0=None):
        # Condition needed to ensure output remains compatible
        system = self.discrete
        # is_ss_input = isinstance(system, StateSpace)

        # Check initial condition
        if x0 is None:
            xout = np.zeros((system.A.shape[1],))
        else:
            xout = np.asarray(x0)

        # Simulate the system
        xout = (np.dot(system.A, xout) +
                            np.dot(system.B, u))
        yout = (np.dot(system.C, xout) +
                        1.0*np.dot(system.D, u))

        return yout, xout

    
    def realify(self, M):
        top_row = np.hstack([M.real, -M.imag])
        bot_row = np.hstack([M.imag,  M.real])
        M_real = np.vstack([top_row, bot_row])
        return M_real


    def qstep(self, input_states, s=None, t=None, x0_means=None, x0_cov=None):
        # Condition needed to ensure output remains compatible
        system = self.discrete
        # is_ss_input = isinstance(system, StateSpace)
        A = self.realify(self.discrete.A)
        B = self.realify(self.discrete.B)
        C = self.realify(self.discrete.C) 
        D = self.realify(self.discrete.D) 


        # means = np.hstack([input_states.means, input_states.means])

        means = input_states.means
        cov = input_states.cov

        # Check initial condition
        if x0_means is None:
            xout_means = np.zeros((A.shape[1],))
        else:
            xout_means = np.asarray(x0_means)

        if x0_cov is None:
            xout_cov = np.zeros((A.shape[0],A.shape[1]))
            I = np.eye(D.shape[0])
            xout_cov = np.linalg.pinv(C)@(0.25*I - D@cov@D.T)@np.linalg.pinv(C.T)
        else:
            xout_cov = np.asarray(x0_cov)



        # Simulate the system
        xout_means = A@xout_means + B@means
        yout_means = C@xout_means + D@means

        # cov = 4 * cov
        xout_cov = A@xout_cov@(A.T) + B@cov@(B.T)
        yout_cov = C@xout_cov@(C.T) + D@cov@(D.T)
       


        # # mean_X = A@xout_means
        # cov_X = A@xout_cov@A.T
        
        # mean_Y = B@means
        # # cov_Y = C@cov@C.T
        # cov_Y = C.T@cov@C

        # # mean_XplusY = mean_X + mean_Y
        # cov_XplusY = cov_X + cov_Y

        # mean_G = C@mean_XplusY
        # cov_G = C@cov_XplusY@C.T

        # mean_H = D@means
        # cov_H = D@cov@D.T

        # # xout_means = mean_G + mean_H 
        # yout_cov = cov_G + cov_H # both this line and a few lines up assume that X, Y / G, H are independent

        # # Calculate the covariances
        # xout_cov = A@xout_cov + B@cov
        # zout_cov = C@xout_cov + D@cov

        # xz0 = A@xz0 + B@((zout_cov).T)
        # yout_cov = (C@xz0 + D@(zout_cov.T)).T

        return yout_means, yout_cov, xout_means, xout_cov

class CVF_Options:
    def __init__(self, 
                 quantum=False,
                 poles_estimation_threshold = 1,
                 model_error_threshold = 1e-6, 
                 max_iterations = 5, 
                 enforce_stability = True, 
                 alpha = 0.01,
                 beta = 2.5,
                 debug = True,
                 mode = "CVF",
                 outlier_threshold = 0.03,
                 real_valued = True,
                 baseband = True,
                 pole_spacing = 'log',
                 dt=1e-15,
                 order=50,
                 center_wvl=1.55):
        self.quantum = quantum
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha
        self.real_valued = real_valued
        self.outlier_threshold = outlier_threshold
        self.baseband = baseband
        self.beta=beta
        self.pole_spacing = pole_spacing
        self.dt = dt
        self.order = order
        self.center_wvl = center_wvl

class CVF_Model:
    def __init__(self, wvl_microns, circuit, options=None):
        if options == None:
            self.options = CVF_Options()
        else:
            self.options = options

        c = 299792458
        # self.order = order
        self.wvl_microns = wvl_microns
        if self.options.baseband == True:
            self.center_freq = c / (self.options.center_wvl * 1e-6)
        else:
            self.center_freq = 0
        self.freqs = np.flip(c / (wvl_microns * 1e-6))
        # self.freqs_shifted = self.freqs
        self.freqs_shifted = self.freqs - self.center_freq
        self.poles = np.array([])
        self.circuit = circuit
        s_params = np.asarray(dict_to_matrix(circuit))

        if self.options.quantum:
            self.S = self.to_unitary(s_params)
        else:
            self.S = s_params

        self.error = float('inf')
        self.num_measurements = self.S.shape[0]
        self.num_outputs = self.S.shape[1]
        self.num_inputs = self.S.shape[2]

        if self.num_inputs == self.num_outputs:
            self.num_ports = self.num_inputs
        else:
            print("Number of Inputs must equal number of outputs")

        # self.D = np.zeros((self.num_outputs, self.num_inputs), dtype=complex)
        # self.residues = np.zeros((order, self.num_outputs, self.num_inputs), dtype=complex)
        # self.time_response = None
        self.num_outliers = 1000


        if isinstance(self.options.order, Iterable):
            for n in options.order:
                print(f"Testing order {n}")
                poles, residues, D, error = self.fit_pole_residue_model(order=n)

                response =  self.compute_model_response(poles, residues, D)
                outliers = np.where(np.abs(np.abs(self.S[:, 1, 0])**2 - np.abs(response[:, 1, 0])**2) > self.options.outlier_threshold)
                # plt.scatter(self.freqs[outliers], np.abs(response[:, 1, 0][outliers])**2)
                # plt.plot(self.freqs, np.abs(response[:, 1, 0])**2)
                # plt.plot(self.freqs, np.abs(self.S[:, 1, 0])**2)
                # plt.show()

                num_outliers = len(outliers[0])

                if error < self.error:
                    self.poles = poles
                    self.residues = residues
                    self.D = D
                    self.error = error
                    self.order = n
                    self.num_outliers = num_outliers
                    # self.outliers = outliers
                    print(f"Order: {n}")
                    print(f"error: {error}")
                    # print(f"Num Outliers: {len(outliers)}")
                    pass
        else:
            self.poles, self.residues,self.D,self.error = self.fit_pole_residue_model(order=self.options.order)
            self.order = self.options.order
            # self.outliers = outliers


        self.state_space = State_Space_Model()
        self.state_space.continuous, self.state_space.discrete = self.compute_state_space_model()
    
    def compute_h(self, t):
        return self.D + np.sum(self.residues*np.exp(self.poles[:, np.newaxis, np.newaxis] * t))
    
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
        

    def compute_state_space_model(self):
        # models = [[None for _ in range(self.num_inputs)] for _ in range(self.num_outputs)]
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

        sys = StateSpace(A, B, C, D)

        return sys, sys.to_discrete(dt=self.options.dt)

    
    def initial_poles(self, order=None):
        if order is None:
            order = self.order
        poles = np.array([])
        
        if self.options.pole_spacing == 'log':
            negative_order = order//2
            positive_order = order//2

            for f in np.logspace(1.0, 0.0, negative_order) * self.freqs_shifted[0]:
                poles = np.append(poles, (self.options.alpha + 1j) *2*np.pi*f)

            if negative_order + positive_order != order:
                poles = np.append(poles, (self.options.alpha + 1j) *2*np.pi*0.0)

            for f in np.logspace(1.0, 0.0, order//2) * self.freqs_shifted[-1]:
                poles = np.append(poles, (-self.options.alpha + 1j) *2*np.pi*f)
        elif self.options.pole_spacing == 'lin':
            beta = self.options.beta
            for f in np.linspace(beta*self.freqs_shifted[0], beta*self.freqs_shifted[-1], order):
                if f.real > 0:
                    poles = np.append(poles, (-self.options.alpha + 1j) *2*np.pi*f)
                else:
                    poles = np.append(poles, (self.options.alpha + 1j) *2*np.pi*f)

        
        # plt.scatter(poles.real, poles.imag)
        # plt.show()

        return poles
        # return (-self.options.alpha + 1j) * np.linspace(2*np.pi*self.freqs_shifted[0], 2*np.pi*self.freqs_shifted[-1], order)
    
    def compute_phi_matrices(self, poles):
        phi1 = 1 / (2*np.pi*1j*self.freqs_shifted[0]-poles)
        for omega in self.freqs_shifted[1:]:
            phi1 = np.vstack((phi1, 1 / (2*np.pi*1j*omega-poles)))
        
        unity_column = np.ones((len(self.freqs_shifted), 1))
        phi0 = np.hstack((unity_column, phi1))

        return phi0, phi1
    
    def compute_lstsq_matrices(self, phi0, phi1):
        D = []
        V = []
        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                D.append(np.diag(self.S[:, i, j]))
                V.append(self.S[:, i, j])

        phi_column = []
        for _D in D:
            phi_column.append(-_D@phi1)

        blocks = [phi0] * (self.num_inputs*self.num_outputs)
        M = block_diag(*blocks)

        phi_column = np.vstack(phi_column)
        M = np.hstack([M, phi_column])

        return M, np.hstack(V)
    
    def compute_model_response(self,poles=None, residues=None, D=None, freqs=None):
        if freqs is None:
            freqs = self.freqs
        if poles is None:
            poles = self.poles
        if residues is None:
            residues = self.residues
        if D is None:
            D = self.D
        freqs_shifted = freqs - self.center_freq
        # freqs_shifted = freqs
        response = np.tile(D, (freqs_shifted.shape[0], 1, 1))
        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                for p, r in zip(poles, residues[:, i, j]):
                    response[:, i, j] = response[:, i, j] + r / (2*np.pi*1j*freqs_shifted - p)
        
        return response
    
    def fit_pole_residue_model(self, order):
        poles = self.initial_poles(order)

        iter = 1
        while iter <= self.options.max_iterations:
            phi0, phi1 = self.compute_phi_matrices(poles)
            M, V = self.compute_lstsq_matrices(phi0, phi1)

            Q,R = np.linalg.qr(M,mode='reduced') 
            # solutions = np.linalg.pinv(R)@Q.conj().T@V

            # solutions, _, _, _ = np.linalg.lstsq(M, V, rcond=None)
            solutions = np.linalg.pinv(R)@Q.conj().T@V
            A = np.diag(poles)
            weight_coefficients = solutions[self.num_outputs*self.num_inputs*(order+1):]
            weights_row = weight_coefficients.reshape((len(weight_coefficients), 1))
            unity_column = np.ones((order, 1))
            poles, _ = np.linalg.eig(A-unity_column@weights_row.T)

            if self.options.enforce_stability:
                mask = np.real(poles) > 0
                poles[mask] = -np.real(poles[mask]) + 1j*np.imag(poles[mask])


            weights = np.ones_like(self.freqs_shifted)
            for w, p in zip(weight_coefficients, poles):
                weights = weights + w / (2*np.pi*1j*self.freqs_shifted - p)

            residues = np.zeros((order, self.num_outputs, self.num_inputs), dtype=complex)
            D = np.zeros((self.num_outputs, self.num_inputs), dtype=complex)
            
            print(f"Estimator: {np.max(np.abs(weights - 1))}")
            if np.max(np.abs(weights - 1)) <= self.options.poles_estimation_threshold or iter >= self.options.max_iterations:
                for i in range(self.num_outputs):
                    for j in range(self.num_inputs):
                        phi0, _ = self.compute_phi_matrices(poles)
                        Q,R = np.linalg.qr(phi0,mode='reduced') 
                        # solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        # solutions, _, _, _ = np.linalg.lstsq(phi0, self.S[:, i, j], rcond=None)
                        D[i, j] = solutions[0]
                        residues[:, i, j] = solutions[1:]

                weighted_error = 0
                absolute_error = 0
                response =  self.compute_model_response(poles, residues, D)
                # outliers = np.where(np.abs(self.S - response) > self.options.outlier_threshold)
                for H, w, H_hat  in zip(self.S, weights,response):
                    weighted_error += np.linalg.norm(H*w - H_hat)
                    # outliers = np.abs(H - H_hat) > self.options.outlier_threshold
                    absolute_error += np.linalg.norm(H - H_hat)

                weighted_error = 1/(self.num_measurements*self.num_outputs*self.num_inputs)*(weighted_error**2)
                absolute_error = 1/(self.num_measurements*self.num_outputs*self.num_inputs)*(absolute_error**2)
                absolute_error = 1/(self.num_measurements*self.num_outputs*self.num_inputs)*(absolute_error**2)
                # print(f'weighted error: {weighted_error}')
                # print(f'absolute error: {absolute_error}')

                if absolute_error < self.options.model_error_threshold:
                    print(f"Success after {iter} iterations")
                    return poles, residues, D, absolute_error


            iter += 1
        
        return poles, residues, D, absolute_error
    
    def compute_steady_state(self, freqs=None):
        if freqs is None:
            freqs = self.freqs
        
        freqs_shifted = freqs - self.center_freq
        steady_state = np.array([])
        for f in freqs_shifted:
            # print(f'{f}')
            A = self.state_space.continuous.A - 2*np.pi*f*1j*np.eye(self.state_space.continuous.A.shape[0])
            B = self.state_space.continuous.B
            C = self.state_space.continuous.C
            D = self.state_space.continuous.D
            sys = StateSpace(A, B, C, D)

            N = int(2)
            T = 20.0e-12
            t = np.linspace(0, T, N)

            sig = np.exp(1j*2*np.pi*t*0)
            sig1 = sig.reshape(-1, 1)
            sigs = np.hstack([1*sig1, 0*sig1, 0*sig1, 0*sig1])

            tout, yout, xout = lsim(sys, sigs, t)
            steady_state = np.append(steady_state, yout[-1, 1])
            # print(f'{steady_state}')
            # plt.title(f"{f}")
            # plt.plot(tout, np.abs(yout[:, 1])**2)
            # plt.show()
        
        plt.plot(freqs, np.abs(steady_state)**2)
        plt.show()
                
    def plot_frequency_domain_model(self):
        freqs = np.linspace(self.freqs[0], self.freqs[-1], 20 * self.S.shape[0])
        fig, ax = plt.subplots(self.num_outputs, self.num_inputs)

        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                ax[i, j].set_title(f'S{i}{j}')
                ax[i, j].plot(freqs, np.abs(self.compute_model_response(freqs=freqs)[:, i, j])**2, label='model')
                ax[i, j].scatter(self.freqs, np.abs(self.S[:, i, j])**2, marker=',', s=5, color='red', alpha=0.25, label='samples')
                ax[i, j].set_xlabel('Frequency Hz')
                ax[i, j].set_ylabel('Transmission')
        ax[0, 0].legend()

        plt.tight_layout()
        plt.show()

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
                 mode = "BVF",
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

class BasebandModelSingleIO:
    def __init__(self, wvl_microns, center_wvl, s_params, order, options=None):
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
        self.S = s_params
        self.error = float('inf')
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.time_response = None

    def initial_poles(self):
        digital_freq = 2 * np.pi * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order)/self.sampling_freq
        #digital_freq = 2 * np.pi * np.linspace(-self.sampling_freq / 2, self.sampling_freq / 2, self.order)
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

        # print(f"Failed to reach poles estimation threshold after {iter} iterations")
        phi0, phi1 = self.compute_phi_matrices()
        self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
        self.error = self.compute_error()
        # self.plot_model()
    
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
        A = np.diag(self.poles)
        U, S, Vh = svd(np.array([[self.residues[1]]]))
        B = Vh
        for n in range(2, self.order + 1):
            # print(n)
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
    
    def compute_state_space_model(self):
        if self.options.real_valued == True:
            self.A, self.B, self.C, self.D = self.real_ABCD_matrices()
        elif self.options.real_valued == False:
            self.A, self.B, self.C, self.D = self.complex_ABCD_matrices()
    
    def compute_steady_state(self):
        pass

    def compute_time_response(self, sig=None, t=None):
        c = 299792458
        if self.A is None:
            self.compute_state_space_model()

        sys = StateSpace(self.A, self.B, self.C, self.D, dt = 1/self.sampling_freq)


        if t is None:
            N = int(1000)
            T = 2e-11
            t = np.linspace(0, T, N)

        if sig is None:
            sig = np.exp(1j*2*np.pi*t*0)
            #sig = np.full(t, 1.0)
        
        sig = sig.reshape(-1, 1)
        impulse = np.hstack([np.real(sig), np.imag(sig)])

        t_out, yout, _ = dlsim(sys, impulse, t)
        yout = yout[:, 0] + 1j*yout[:, 1]
        self.time_response = (t_out, yout)

        return t_out, yout
        

    def plot_time_response(self):
        if self.time_response is None:
            self.compute_time_response

        t_out, yout = self.time_response

        plt.title("Time Response")
        plt.xlabel("Time")
        plt.ylabel("E-field Amplitude")
        # plt.plot(t_out, np.abs(yout[:, 0] + 1j*yout[:, 1])**2)
        plt.plot(t_out, np.abs(yout)**2)
        plt.axvline(x=0.59e-12, color='r', linestyle='--', linewidth=1, alpha=0.75)



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

class BasebandModel:
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
        # self.sampling_freq = -self.options.beta * (np.max(self.freqs) - np.min(self.freqs))
        self.sampling_freq = self.options.beta * (self.freqs[-1] - self.freqs[0])

        # self.freqs = np.flip(c / (wvl_microns * 1e-6)) - self.center_freq
        # self.sampling_freq = self.options.beta * (np.max(self.freqs) - np.min(self.freqs))

        self.poles = np.array([])
        self.residues = np.zeros((order, self.num_ports, self.num_ports), dtype=complex)

        self.digital_freq = 2 * np.pi * self.freqs / (self.sampling_freq)
        # self.digital_freq = 2 * np.pi * np.linspace(np.min(self.freqs), np.max(self.freqs), self.order) / self.sampling_freq
        self.z = np.exp(1j*self.digital_freq)
        # self.S = np.flip(s_params)
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
        # digital_freq = 2 * np.pi * np.linspace(np.max(self.freqs), np.min(self.freqs), self.order)/self.sampling_freq
        return self.options.gamma * np.exp(1j*digital_freq)
    
    def compute_phi_matrices(self):
        phi1 = 1/(self.z[0]-self.poles)
        for z in self.z[1:]:
            phi1 = np.vstack((phi1, 1 / (z-self.poles)))
        
        unity_column = np.ones((len(self.z), 1))
        phi0 = np.hstack((unity_column, phi1))

        return phi0, phi1
    
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

    def fit_model(self):
        self.poles = self.initial_poles()

        iter = 1
        while iter < self.options.max_iterations:
            phi0, phi1 = self.compute_phi_matrices()
            M, V = self.compute_lstsq_matrices(phi0, phi1)
            Q,R = np.linalg.qr(M,mode='reduced') 
            solutions = np.linalg.pinv(R)@Q.conj().T@V
            # inital_residues = solutions[:self.order+1]
            # self.weight_coefficients = solutions[self.order+1:]

            # Calculate New Poles
            A = np.diag(self.poles)
            weight_coefficients = solutions[(self.num_ports**2)*(self.order+1):]
            weights_row = weight_coefficients.reshape((len(weight_coefficients), 1))
            unity_column = np.ones((self.order, 1))
            self.poles, _ = np.linalg.eig(A-unity_column@weights_row.T)
            mask = np.abs(self.poles) > 1
            self.poles[mask] = 1 / (self.poles[mask])

            if True:
                # self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
                for i in range(self.num_ports):
                    for j in range(self.num_ports):
                        phi0, _ = self.compute_phi_matrices()
                        Q,R = np.linalg.qr(phi0,mode='reduced') 
                        # solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
                        # solutions, _, _, _ = np.linalg.lstsq(phi0, self.S[:, i, j], rcond=None)
                        self.D[i, j] = solutions[0]
                        self.residues[:, i, j] = solutions[1:]

            #weighting_term = 1 + self.weight_coefficients@phi1
            # self.weights = self.calculate_weights(phi1)
            #print(np.max(np.abs(weighting_term-1)))

            # if np.max(np.abs(self.weights-1)) < self.options.poles_estimation_threshold:
            #     self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
            # unity_column = np.ones((self.order, 1))
            # A = np.diag(self.poles)
            # #bw = unity_column
            # weights_row = self.weight_coefficients.reshape((len(self.weight_coefficients), 1)).T
            # self.poles, _ = np.linalg.eig(A-unity_column@weights_row)
            iter += 1

        # print(f"Failed to reach poles estimation threshold after {iter} iterations")
        # phi0, phi1 = self.compute_phi_matrices()
        # self.residues, _, _, _ = np.linalg.lstsq(phi0, V, rcond=None)
        # self.error = self.compute_error()
        # self.plot_model()
    
    def calculate_weights(self, phi1):
        return phi1@self.weight_coefficients
    
    def compute_response(self, a, b, wvl=None):
        if wvl is None:
            wvl = self.wvl_microns

        c = 299792458
        # freq = np.flip(c / (wvl * 1e-6)) - self.center_freq
        freq = c / (wvl * 1e-6) - self.center_freq
        digital_freq = 2 * np.pi * freq / self.sampling_freq
        z = np.exp(1j*digital_freq)
        # response = np.full_like(z, self.residues[0], dtype=complex)
        response = np.stack([self.D]*(z.shape[0]))
        for r, p in zip(self.residues[:, b, a], self.poles[:]):
            response[:, b, a] += r / (z - p)        

        # if freq is not None:
        #     digital_freq = 2 * np.pi * freq / self.sampling_freq
        #     z = np.exp(1j*digital_freq)
        #     response = np.full_like(z, self.residues[0], dtype=complex)
        #     for r, p in zip(self.residues[1:], self.poles):
        #         response += + r / (z - p)        
        # elif wvl is not None:
        #     freq = np.flip(c / (wvl * 1e-6)) - self.center_freq
        #     digital_freq = 2 * np.pi * freq / self.sampling_freq
        #     z = np.exp(1j*digital_freq)
        #     response = np.full_like(z, self.residues[0], dtype=complex)
        #     for r, p in zip(self.residues[1:], self.poles):
        #         response += + r / (z - p)        
        # else:
        #     freq = np.flip(c / (self.wvl_microns * 1e-6)) - self.center_freq
        #     digital_freq = 2 * np.pi * freq / self.sampling_freq
        #     z = np.exp(1j*digital_freq)
        #     response = np.full_like(z, self.residues[0], dtype=complex)
        #     for r, p in zip(self.residues[1:], self.poles):
        #         response += + r / (z - p)        

        # response1 = np.zeros((len(z)), dtype=complex)
        # for i in range(len(z)):
        #     response1[i] = self.residues[0] 
        #     for n in range(self.order):
        #         response1 += self.residues[n+1] / (z[i] - self.poles[n])

        
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
            # fig, ax = plt.subplots(n, 1, figsize=(10, 10))
            for mode in modes:
                plt.plot(np.abs(self.compute_response(mode[0], mode[1]))**2)
                plt.plot(np.abs(self.S[:, mode[0], mode[1]])**2, "r--")
                plt.show()
        
    
    def compute_error(self):
        return np.max(np.abs(self.S - self.compute_response()))

    def complex_ABCD_matrices(self):
        A = np.diag(self.poles)
        U, S, Vh = svd(np.array([[self.residues[1]]]))
        B = Vh
        for n in range(2, self.order + 1):
            # print(n)
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
    
    def compute_state_space_model(self):
        if self.options.real_valued == True:
            self.A, self.B, self.C, self.D = self.real_ABCD_matrices()
        elif self.options.real_valued == False:
            self.A, self.B, self.C, self.D = self.complex_ABCD_matrices()
    
    def compute_steady_state(self):
        pass

    def compute_time_response(self, sig=None, t=None):
        c = 299792458
        if self.A is None:
            self.compute_state_space_model()

        sys = StateSpace(self.A, self.B, self.C, self.D, dt = 1/self.sampling_freq)


        if t is None:
            N = int(1000)
            T = 2e-11
            t = np.linspace(0, T, N)

        if sig is None:
            sig = np.exp(1j*2*np.pi*t*0)
            #sig = np.full(t, 1.0)
        
        sig = sig.reshape(-1, 1)
        impulse = np.hstack([np.real(sig), np.imag(sig)])

        t_out, yout, _ = dlsim(sys, impulse, t)
        yout = yout[:, 0] + 1j*yout[:, 1]
        self.time_response = (t_out, yout)

        return t_out, yout
        

    def plot_time_response(self):
        if self.time_response is None:
            self.compute_time_response

        t_out, yout = self.time_response

        plt.title("Time Response")
        plt.xlabel("Time")
        plt.ylabel("E-field Amplitude")
        # plt.plot(t_out, np.abs(yout[:, 0] + 1j*yout[:, 1])**2)
        plt.plot(t_out, np.abs(yout)**2)
        plt.axvline(x=0.59e-12, color='r', linestyle='--', linewidth=1, alpha=0.75)



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

    


    
# class Damp():
#     def __init__(self, system, t, pulse_amp, n, num_modes):
#         input_signal = np.zeros((t.shape[0], num_modes), dtype=complex)
#         input_signal[:, n] = self.gaussian_pulse(t, 50*system.dt, 20*system.dt)
#         _, yout, _ = self.dlsim_complex(system, input_signal)
#         self.delays = []
#         self.amps = []
#         self.theta = []
    
#     @staticmethod
#     def gaussian_pulse(t, t0, std, a=1.0 ):
#         return a * jnp.exp(-(t - t0)**2 / std**2) 

#     @staticmethod
#     def dlsim_complex(system, u, t=None, x0=None):
#         out_samples = len(u)
#         stoptime = (out_samples - 1) * system.dt

#         xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
#         yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
#         tout = np.linspace(0.0, stoptime, num=out_samples)

#         xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)

#         u_dt = u

#         # Simulate the system
#         for i in range(0, out_samples - 1):
#             xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
#                             np.dot(system.B, u_dt[i, :]))
#             yout[i, :] = (np.dot(system.C, xout[i, :]) +
#                         np.dot(system.D, u_dt[i, :]))

#         # Last point
#         yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
#                                 np.dot(system.D, u_dt[out_samples-1, :]))

#         return tout, yout, xout

# class DampModel():
#     """Damp Model

#     Parameters
#     ----------
#     ckt : sax.saxtypes.Model
#         The circuit to simulate.
#     wl : ArrayLike
#         The array of wavelengths to simulate (in microns).
#     **params
#         Any other parameters to pass to the circuit.

#     Examples
#     --------
#     >>> sim = QuantumSim(ckt=mzi, wl=wl, top={"length": 150.0}, bottom={"length": 50.0})
#     """

#     def __init__(self, baseband_model: BasebandModel, T: float):
#         self.system = baseband_model.generate_sys_discrete()
#         self.dt = self.system.dt
#         self.num_modes = self.system.B.shape[1]
#         self.T = T
#         self.K = round(self.T / self.dt)
#         self.t = np.linspace(0.0, self.K*self.dt, self.K)
#         self.damps = []
    
#     def calculate_damps(self, sig, discretized=True, dy=0.0001):
#         t_orig = np.linspace(0.0, self.T, sig.shape[0])
#         new_sig = np.zeros((self.K, self.num_modes), dtype=complex)
#         for n in range(self.num_modes):
#             new_sig[:, n] = np.interp(self.t, t_orig, sig[:, n])
#         if discretized:
#             real_part = np.round(new_sig.real/dy)*dy
#             imag_part = np.round(new_sig.imag/dy)*dy
#             new_sig = real_part + 1j*imag_part
        
#         for n in range(self.num_modes):
#             _damps = {}
#             for y in new_sig[:, n]:
#                 print(y)
#                 if y not in _damps:
#                     _damps[y] = Damp(self.system, self.t, y, n, self.num_modes) # needs replacing
#             self.damps.append(_damps)
        
#         plt.plot(new_sig)
#         pass

    

