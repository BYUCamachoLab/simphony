from scipy.linalg import block_diag
import numpy as np
from numpy.linalg import svd
from scipy.signal import  StateSpace, dlsim, lsim
from simphony.utils import dict_to_matrix
from collections.abc import Iterable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from simphony.time_domain.time_system import TimeSystem

class PoleResidueModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def plot_poles(self)->None:
        pass

    @abstractmethod
    def to_time_system(self)->TimeSystem:
        #switch to time system
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


    def ABCD_matrices(self):
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

    def compute_state_space_model(self):
        self.A, self.B, self.C, self.D = self.ABCD_matrices()


    # def compute_phi_matrices(self, x, poles):
    #         phi1 = 1 / (x[0]-poles)
    #         for omega in x[1:]:
    #             phi1 = np.vstack((phi1, 1 / (omega-poles)))
            
    #         unity_column = np.ones((len(x), 1))
    #         phi0 = np.hstack((unity_column, phi1))

    #         return phi0, phi1


    def compute_error(self):
            return np.max(np.abs(self.S - self.compute_response()))

class CVFBaseband_Options:
     def __init__(self,
                poles_estimation_threshold = 1,
                model_error_threshold = 1e-10, 
                 max_iterations = 5, 
                 enforce_stability = True, 
                 alpha = 0.01,
                 beta = 2.5,
                 debug = True,
                 real_valued = True,
                 pole_spacing = 'log',
                 dt=1e-15,
                 center_wvl=1.55) :
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha
        self.real_valued = real_valued
        self.beta=beta
        self.pole_spacing = pole_spacing
        self.dt = dt
        self.center_wvl = center_wvl


class CVFModel(PoleResidueModel):
    def __init__(self) -> None: 
        pass




class CVFModelBaseband(PoleResidueModel):
    def __init__(self, wvl_microns, circuit, order, options = None):
        if options == None:
            self.options = CVF_Options()
        else:
            self.options = options

        c = 299792458
        # self.order = order
        self.order = order
        self.wvl_microns = wvl_microns
        if self.options.baseband == True:
            self.center_freq = c / (self.options.center_wvl * 1e-6)
        else:
            self.center_freq = 0
        self.freqs = np.flip(c / (wvl_microns * 1e-6))
        
        self.freqs_shifted = self.freqs - self.center_freq
        
        self.circuit = circuit
        s_params = np.asarray(dict_to_matrix(circuit))
        self.S = s_params

        self.error = float('inf')
        self.num_measurements = self.S.shape[0]
        self.num_outputs = self.S.shape[1]
        self.num_inputs = self.S.shape[2]

        if self.num_inputs == self.num_outputs:
            self.num_ports = self.num_inputs
        else:
            print("Number of Inputs must equal number of outputs")

       
        
        if isinstance(self.order, Iterable):
            for n in self.order:
                print(f"Testing order {n}")
                poles, residues, D, error = self.fit_pole_residue_model(order=n)

                if error < self.error:
                    self.poles = poles
                    self.residues = residues
                    self.D = D
                    self.error = error
                    self.order = n
                    
                    print(f"Order: {n}")
                    print(f"error: {error}")
                    
                    pass
        else:
            self.poles, self.residues,self.D,self.error = self.fit_pole_residue_model(order=self.order)
            self.order = self.order
        

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
        return poles

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

    def compute_phi_matrices(self, poles):
        phi1 = 1 / (2*np.pi*1j*self.freqs_shifted[0]-poles)
        for omega in self.freqs_shifted[1:]:
            phi1 = np.vstack((phi1, 1 / (2*np.pi*1j*omega-poles)))
        
        unity_column = np.ones((len(self.freqs_shifted), 1))
        phi0 = np.hstack((unity_column, phi1))

        return phi0, phi1

        





class IIRModelBaseband(PoleResidueModel):
    def __init__(self) -> None:
        pass


