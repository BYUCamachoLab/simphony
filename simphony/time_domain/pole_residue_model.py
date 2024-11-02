from scipy.linalg import block_diag
import numpy as np
from numpy.linalg import svd
from scipy.signal import  StateSpace, dlsim, lsim


class PoleResidueModel():
    def __init__(self) -> None:
        pass

    
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


    def compute_phi_matrices(poles, x):
            phi1 = 1 / (x[0]-poles)
            for omega in x[1:]:
                phi1 = np.vstack((phi1, 1 / (x-poles)))
            
            unity_column = np.ones((len(x), 1))
            phi0 = np.hstack((unity_column, phi1))

            return phi0, phi1


    def compute_error(self):
            return np.max(np.abs(self.S - self.compute_response()))


class CVFModel(PoleResidueModel):
    def __init__(self) -> None:
        pass

class CVFModelBaseband(PoleResidueModel):
    def __init__(self) -> None: 
        pass


class IIRModelBaseband(PoleResidueModel):
    def __init__(self) -> None:
        pass


