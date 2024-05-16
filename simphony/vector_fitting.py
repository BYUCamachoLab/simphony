import numpy as np
from simphony.libraries import siepic
from simphony.utils import dict_to_matrix
from scipy.linalg import block_diag
import pandas as pd
import math
import cmath

c = 299792458


def is_even(x):
    return x % 2 == 0

def initialize_poles(freq_THz, order, alpha = 0.01):
    poles = np.zeros(order, dtype=complex)

    for n in range(order // 2):
        poles[n] = (-alpha + 1j) * ( freq_THz[0] + (freq_THz[-1] - freq_THz[0]) / (order / 2 - 1) * ((n+1)-1))

    for n in range(order // 2, order):
        poles[n] = np.conj(poles[n - order // 2])

    return poles 


def frobenius_norm(matrix):
    pass

# Solves for x in equations of the form Ax = b 
# A is the coefficient matrix
# b is the constant vector
def least_squares(coefficient_matrix, constant_vector):
    df = pd.DataFrame(coefficient_matrix)
    df.to_csv("cm_ls.csv")

    q, r = np.linalg.qr(coefficient_matrix)
    return np.linalg.pinv(r) @ np.matrix.transpose(q) @ constant_vector

def new_poles(poles, c_w):
    n = np.shape(poles)[0]
    b = np.ones((n, 1), dtype=complex)    
    eigenvalues, _ = np.linalg.eig(np.diag(poles) - b @ np.transpose(c_w[:, np.newaxis]))
    if len(eigenvalues) != n:
        pass
    return eigenvalues


def enforce_pole_stability(poles):
    for p in range(np.shape(poles)[0]):
        if np.real(poles[p]) > 0:
            poles[p] = -np.real(poles[p]) + 1j * np.imag(poles[p])
    return poles
    


def print_matrix(A):
    rows = np.shape(A)[0]
    cols = np.shape(A)[1]
    for row in range(rows):
        print('\n')
        for col in range(cols):
            if A[row, col] != 0+0j:
                print(1, end='')
            else:
                print(0, end='')

def stable_poles(weights, epsilon_w):
    print(f"Pole Stability: {np.abs(max(weights - 1))}")
    return np.abs(max(weights - 1)) <= epsilon_w

def calculate_weights(c_w, poles, freq_THz, order, k_max):
    weights = np.zeros((k_max), dtype=complex)
    for k in range(k_max):
        weights[k] = 1 
        for n in range(order):
            weights[k] += c_w[n] / (1j*freq_THz[k] - poles[n])
    
    return weights

def estimate_H(poles, residues, freq_THz, order, k_max, q_max, m_max):
    H_hat = np.zeros((k_max, q_max, m_max), dtype=complex)
    for k in range(k_max):
        for q in range(q_max):
            for m in range(m_max):
                H_hat[k, q, m] += residues[0, q, m] 
                for n in range(order):
                    H_hat[k, q, m] += residues[n + 1, q, m] / (1j * freq_THz[k] - poles[n])
    
    return H_hat

def final_fitting(Psi_0, V_H, k_max, q_max, m_max):
    pass


def get_weights_and_residues(coefficient_matrix, constant_vector, poles, freq_THz, order, k_max, q_max, m_max ):
    solutions = least_squares(coefficient_matrix= coefficient_matrix, constant_vector=constant_vector)
    #c_H = solutions[:order * q * m]
    c_w = solutions[(order + 1)*q_max*m_max:]

    # residues have shape [order, q, m]
    residues = np.zeros((order+1, q_max, m_max), dtype=complex)
    for q in range(q_max):
        for m in range(m_max):
            for n in range(order + 1):
                index = (q * m_max + m) * (order + 1) + n 
                residues[n, q, m] = solutions[index]
    

    weights = calculate_weights(c_w=c_w, poles=poles, freq_THz=freq_THz, order=order, k_max=k_max)
    return weights, residues




def compute_fitting_error(H, H_hat, weights, k_max, q_max, m_max):
    error = 0
    for k in range(k_max):
        error += np.linalg.norm(H[k] * weights[k] - H_hat[k], ord='fro')**2
    
    error *= 1 / (k_max * q_max * m_max)
    return error



def vector_fitting(H, freq_THz, order = 10, max_iterations = 100, epsilon_w =0.5): 
    if is_even(order) == False:
        print("Order must be even")
        return None

    k_max = np.shape(H)[0] # Number of measurements
    q_max = np.shape(H)[1] # Number of outputs
    m_max = np.shape(H)[2] # Number of inputs



    poles = initialize_poles(freq_THz = freq_THz, order = order)

    c_H = np.zeros((order+1, q_max, m_max), dtype=complex)
    c_w = np.zeros((order), dtype=complex)
    Psi_0 = np.zeros((k_max, order + 1), dtype=complex)
    Psi_1 = np.zeros((k_max, order), dtype=complex)

    D_H = np.zeros((k_max, k_max, q_max, m_max), dtype=complex)
    V_H = np.zeros((k_max, q_max, m_max), dtype=complex)


    # initialize D_H and V_H
    _V_H = []
    for q in range(q_max):
        for m in range(m_max):
            for k in range(k_max):
                D_H[k, k, q, m] = H[k, q, m]
                _V_H.append(H[k, q, m])
    
    V_H = np.array(_V_H)


    i = 1
    while i <= max_iterations:
        print(f"Iteration: {i}")
        # Update Psi_0
        # for k in range(k_max):
        #     for n in range(order):
        #         if n == 0:
        #             Psi_0[k, n] = 1
        #         else:
        #             omega = freq_THz[k]
        #             p = poles[n-1]
        #             Psi_0[k, n] = 1 / (1j*omega - p)
        
        # Update Psi_1 and Psi_0
        for k in range(k_max):
            for n in range(order):
                omega = freq_THz[k]
                p = poles[n]
                #if not math.isclose(np.real(1j*omega), np.real(p), abs_tol=0.001) and not math.isclose(np.imag(1j*omega), np.imag(p), abs_tol=0.001):
                if not cmath.isclose(1j*omega, p, abs_tol=0.00001):
                    Psi_1[k, n] = 1 / (1j*omega - p)

        ones_column = np.ones(k_max).reshape(k_max, 1)
        Psi_0 = np.hstack((ones_column, Psi_1))

        # Update coefficient matrix
        #coefficient_matrix_old = np.zeros((k_max*q_max*m_max, (order + 1)*q_max*m_max + order), dtype=complex)
        #constant_vector = np.zeros((k_max*q_max*m_max), dtype=complex)

        # Update coefficient matrix with Psi_0 values
        # j_max = q_max * m_max
        # for j in range(j_max):
        #     for k in range(k_max):
        #         for n in range(order + 1):
        #             coefficient_matrix_old[j*k_max + k, (order + 1)*j + n] = Psi_0[k,n]

        # Update coeffiecient matrix with -D_H[q][m] @ Psi_1 values
        # U = np.zeros((k_max, order), dtype=complex)
        # for q in range(q_max):
        #     for m in range(m_max):
        #         for k in range(k_max):
        #             for n in range(order):
        #                 U = -D_H[:, :, q, m] @ Psi_1
        #                 coefficient_matrix_old[(q * m_max + m)*k_max + k, (order + 1) * j_max + n] = U[k, n]
        #                 # coefficient_matrix[(q * m_max + m)*k_max + k, (order + 1) * j_max + n] = 1
        
        # coefficient_matrix = np.hstack((Psi_0, -D_H @ Psi_1))
        Psi_0_blocks = [Psi_0] * q_max * m_max
        Psi_0_diag = block_diag(*Psi_0_blocks)
        
        DPsi_1_blocks = []
        for q in range(q_max):
            for m in range(m_max):
                DPsi_1_blocks.append(-D_H[:, :, q, m] @ Psi_1)
        
        DPsi_1_column = np.vstack(DPsi_1_blocks)
        coefficient_matrix = np.hstack((Psi_0_diag, DPsi_1_column))


        df = pd.DataFrame(coefficient_matrix)
        df.to_csv("cm.csv")


        #print_matrix(coefficient_matrix)

        # _V_H = []
        # for q in range(q_max):
        #     for m in range(m_max):        
        #         _V_H.append(V_H[:, q,m])
        
        constant_vector = V_H
        # weights, residues = get_weights_and_residues(coefficient_matrix, constant_vector, poles, freq_THz, order, k_max, q_max, m_max)

        solutions = least_squares(coefficient_matrix= coefficient_matrix, constant_vector=constant_vector)
        #c_H = solutions[:order * q * m]
        c_w = solutions[(order + 1)*q_max*m_max:]

        # residues have shape [order, q, m]
        residues = np.zeros((order+1, q_max, m_max), dtype=complex)
        for q in range(q_max):
            for m in range(m_max):
                for n in range(order + 1):
                    index = (q * m_max + m) * (order + 1) + n 
                    residues[n, q, m] = solutions[index]
    

        weights = calculate_weights(c_w=c_w, poles=poles, freq_THz = freq_THz, order=order, k_max=k_max)
        
        H_hat = estimate_H(poles = poles, residues = residues, freq_THz = freq_THz, order = order, k_max = k_max, q_max = q_max, m_max = m_max)

        poles = new_poles(poles=poles, c_w = c_w)
        # poles = enforce_pole_stability(poles=poles)

        if stable_poles(weights = weights, epsilon_w = epsilon_w):
            fitting_error = compute_fitting_error(H, H_hat, weights, k_max, q_max, m_max)

            # Update Psi_1 and Psi_0
            for k in range(k_max):
                for n in range(order):
                    omega = freq_THz[k]
                    p = poles[n]
                    if not cmath.isclose(1j*omega, p, abs_tol=0.00001):
                        Psi_1[k, n] = 1 / (1j*omega - p)

            ones_column = np.ones(k_max).reshape(k_max, 1)
            Psi_0 = np.hstack((ones_column, Psi_1))

            for q in range(q_max):
                for m in range(m_max):
                    index = (q * m_max + m) * (k_max)
                    c_H[:, q, m] = least_squares(Psi_0, V_H[index:index+k_max])
            
            residues = c_H


            # Using this Psi_0, we will perform a step similar to the last least squares process except, 
            # We solve each input/output individually and we do not need a weighting term
            
            return poles, residues, fitting_error

        i += 1

import matplotlib.pyplot as plt

def convert_pole_residue_form(poles, residues, x_vals):
    pass


if __name__ == '__main__':
    # section 3.3 example
    # pole = [-1.3578, -1.2679, -1.4851 + 0.2443*1j,-1.4851 - 0.2443*1j, -0.8487 + 2.9019*1j,-0.8487 - 2.9019*1j, -0.8587+3.1752*1j,-0.8587-3.1752*1j, -0.2497+6.5369*1j, -0.2497-6.5369*1j]
    # residue = [0.1059, -0.2808, 0.1166, 0.9569 - 0.7639*1j,0.9569 + 0.7639*1j, 0.9357 - 0.7593 * 1j,0.9357 + 0.7593 * 1j, 0.4579-0.7406*1j,0.4579+0.7406*1j, 0.2405-0.7437*1j, 0.2405+0.7437*1j]

    # order = len(pole)

    # x = np.linspace(0, 10, 100)
    # y = []
    # for omega in x:
    #     flow_rate = residue[0]
    #     for n in range(order):
    #         flow_rate += residue[n+1] / (1j * omega - pole[n])
    #     y.append(flow_rate)

    # print(f"Flow rate mag {np.absolute(flow_rate)}")

    #plt.plot(x, np.absolute(y))
    #plt.show()

    #plt.plot(x, np.arctan(np.imag(y)/np.real(y)) * 180 / np.pi)
    #plt.show()
    wvl = np.linspace(1.5, 1.6, 100)
    
    freq_THz = (c / wvl) * 1e-6
    omega_Trad_per_sec = 2 * np.pi * freq_THz
    omega =  omega_Trad_per_sec * 1e+12
    s = siepic.directional_coupler(wl=wvl)
    H = dict_to_matrix(s)
    order=22

    num_samples = 100
    x = omega_Trad_per_sec / 100

    poles, residues, fitting_error = vector_fitting(H = H, freq_THz = x, order=order, max_iterations=10000, epsilon_w=1)
    H_hat = estimate_H(poles, residues, x, order, num_samples, 4, 4)

    y = np.absolute(H_hat[:, 0, 0])
    plt.plot(x / (2 * np.pi), y, label = "Model")
    plt.plot(x / (2 * np.pi), np.absolute(H[:, 0, 0]), label = "Samples")
    plt.legend()
    plt.show()

    y = np.absolute(H_hat[:, 0, 1])
    plt.plot(x / (2 * np.pi), y, label = "Model")
    plt.plot(x / (2 * np.pi), np.absolute(H[:, 1, 0]), label = "Samples")
    plt.legend()
    plt.show()

    y = np.absolute(H_hat[:, 1, 0])
    plt.plot(x / (2 * np.pi), y, label = "Model")
    plt.plot(x / (2 * np.pi), np.absolute(H[:, 1, 0]), label = "Samples")
    plt.legend()
    plt.show()

    y = np.absolute(H_hat[:, 1, 1])
    plt.plot(x / (2 * np.pi), y, label = "Model")
    plt.plot(x / (2 * np.pi), np.absolute(H[:, 1, 1]), label = "Samples")
    plt.legend()
    plt.show()

    y_hat = H_hat[:, 1, 0]
    y = H[:, 1, 0]
    plt.plot(x / (2 * np.pi), np.arctan(np.imag(y_hat)/np.real(y_hat)) * 180 / np.pi, label = "Model")
    plt.plot(x / (2 * np.pi), np.arctan(np.imag(y)/np.real(y)) * 180 / np.pi, label = "Samples")
    plt.legend()
    plt.show()
    pass



    # wvl = np.linspace(1.5, 1.6, 100)
    
    # freq_THz = (c / wvl) * 1e-6
    # omega_Trad_per_sec = 2 * np.pi * freq_THz
    # omega =  omega_Trad_per_sec * 1e+12
    # s = siepic.directional_coupler(wl=wvl)
    # H = dict_to_matrix(s)
    # order=10

    # num_samples = 100
    # x = omega_Trad_per_sec

    # poles, residues, fitting_error = vector_fitting(H = H, freq_THz = omega_Trad_per_sec, order=order, epsilon_w=1.5)
    # H_hat = estimate_H(poles, residues, x, order, num_samples, 4, 4)
    # y = np.absolute(H_hat[:, 2, 1])
    # plt.plot(x / (2 * np.pi), y)
    # plt.plot(x / (2 * np.pi), np.absolute(H[:, 2, 1]))
    # plt.show()


    # 5 measurements, 1 outputs, 2 inputs 
    # test_H = np.zeros((4, 2, 2), dtype=complex)
    # test_H[:, 0, 0] = np.array([3.5 +.5j,  4.6 - .9j,  5.9 + .4j,  6.5 + 1j])
    # test_H[:, 0, 1] = np.array([1.0 -1j,  4 - .8j,  5 + 1j,  6 + 7j])
    # test_H[:, 1, 0] = np.array([1.0 -1j,  4 - .8j,  5 + 1j,  6 + 7j])
    # test_H[:, 1, 1] = np.array([1.0 -1j,  4 - .8j,  5 + 1j,  6 + 7j])
    # omega = [4.9, 6.04, 7.3, 8.1]
    # order = 16

    # poles, residues, fitting_error = vector_fitting(H = test_H, freq_THz = omega, order=order, epsilon_w=0.2)
    # k_max = 1000
    # x = np.linspace(5, 8, k_max)
    # H_hat = estimate_H(poles, residues, x, order, k_max, 2, 2)

    # plt.plot(x, np.absolute(H_hat[:, 0, 1]))
    # plt.show()