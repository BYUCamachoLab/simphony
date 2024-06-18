# P. Triverio, "Vector Fitting", in P. Benner, S. Grivet-Talocia, A. Quarteroni, G. Rozza, W. H. A. Schilders, L. M. Silveira (Eds.), "Model Order Reduction. Volume 1: System- and Data-Driven Methods and Algorithms", De Gruyter, 2021.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import block_diag
import pandas as pd

from simphony.libraries import siepic
from simphony.utils import dict_to_matrix
from scipy.signal import StateSpace, bilinear_zpk

from numpy.linalg import svd

def samples_flow_rate(omega):
    # section 3.3 example
    pole = [-1.3578, -1.2679, -1.4851 + 0.2443*1j,-1.4851 - 0.2443*1j, -0.8487 + 2.9019*1j,-0.8487 - 2.9019*1j, -0.8587+3.1752*1j,-0.8587-3.1752*1j, -0.2497+6.5369*1j, -0.2497-6.5369*1j]
    residue = [0.1059, -0.2808, 0.1166, 0.9569 - 0.7639*1j,0.9569 + 0.7639*1j, 0.9357 - 0.7593 * 1j,0.9357 + 0.7593 * 1j, 0.4579-0.7406*1j,0.4579+0.7406*1j, 0.2405-0.7437*1j, 0.2405+0.7437*1j]
    y = []
    order = len(pole)
    for s in omega * 1j:
        flow_rate = residue[0]
        for n in range(order):
            flow_rate += residue[n+1] / (s - pole[n])
        y.append(flow_rate)
    
    return y

class Pole_Residue_Model:
    def __init__(self):
        pass

class VF_Options:
    def __init__(self, poles_estimation_threshold = 1e-1, 
                 model_error_threshold = 1e-3, 
                 max_iterations = 5, 
                 enforce_stability = True, 
                 alpha = 0.01,
                 debug = True,
                 mode = "CVF"):
        self.poles_estimation_threshold = poles_estimation_threshold
        self.model_error_threshold = model_error_threshold
        self.max_iterations = max_iterations
        self.enforce_stability = enforce_stability
        self.debug = debug
        self.alpha = alpha

def compute_phi_matrices(omega,poles_real,poles_complex):
    kbar = len(omega)
    if np.shape(poles_real) == ():
        num_poles_real = 0
    else:
        num_poles_real = len(poles_real)

    num_complex_pairs = len(poles_complex)

    phi_real = np.zeros((kbar, num_poles_real), dtype=complex)
    for i in range(num_poles_real):
        phi_real[:, i] = 1. / (1j*omega[:, 0]-poles_real[i])
    phi_complex = np.zeros((kbar, 2 * num_complex_pairs), dtype=complex)
    for i in range(num_complex_pairs):
        phi_complex[:, 2*i] = 1. / (1j*omega[:, 0] - poles_complex[i]) + 1. / (1j*omega[:, 0] - np.conj(poles_complex[i]))
        phi_complex[:, 2*i + 1] = 1j / (1j*omega[:, 0] - poles_complex[i]) - 1j / (1j*omega[:, 0] - np.conj(poles_complex[i]))



    return phi_real, phi_complex

def ComputeModelResponse(omega, R0, Rr, Rc, pr, pc):
    """
    Compute the frequency response of a Vector Fitting model

    Parameters:
    - omega: frequency samples, column vector. This is angular frequency (omega = 2*pi*f)
    - R0: constant coefficient
    - Rr: residues of real poles, 3D array. First dimension corresponds to system outputs. 
          Second dimension to system inputs. Third dimension corresponds to the various poles.
    - Rc: residues of complex conjugate pole pairs (only one per pair)
    - pr: real poles, column vector
    - pc: complex poles, column vector. Only one per pair of conjugate poles

    Returns:
    - H: model response samples, 3D array. First dimension corresponds to system outputs.
         Second dimension to system inputs. Third dimension corresponds to frequency.
    """
    qbar = R0.shape[0]     # number of outputs
    mbar = R0.shape[1]     # number of inputs
    kbar = len(omega)      # number of frequency points

    nr = len(pr)           # number of real poles
    nc = len(pc)           # number of complex conjugate pairs

    # Preallocate space for H
    H = np.zeros((qbar, mbar, kbar), dtype=complex)

    # Compute the model response
    for ik in range(kbar):
        H[:, :, ik] = R0
        for ir in range(nr):
            H[:, :, ik] += Rr[:, :, ir] / (1j * omega[ik] - pr[ir])
        for ic in range(nc):
            H[:, :, ik] += Rc[:, :, ic] / (1j * omega[ik] - pc[ic])
            H[:, :, ik] += np.conj(Rc[:, :, ic]) / (1j * omega[ik] - np.conj(pc[ic]))

    return H

def compute_time_response(times, R0, Rr, Rc, pr, pc):
    nr = len(pr)
    nc = len(pc)
    q_bar = R0.shape[0]
    m_bar = R0.shape[1]

    h = np.zeros((len(times), q_bar, m_bar), dtype=complex)

    for t_index in range(len(times)):
        t = times[t_index]

        h[t_index, :, :] = R0
        # real poles
        for n in range(nr):
            h[t_index, :, :] += Rr[:, :, n] * np.exp(pr[n] * t)

        # complex poles
        for n in range(nc):
            h[t_index, :, :] += 2 * np.real(Rc[:, :, n]) * np.exp(np.real(pc[n]) * t) * np.cos(np.imag(pc[n]) * t) - 2 * np.imag(Rc[:, :, n]) * np.exp(np.real(pc[n]) * t) * np.sin(np.imag(pc[n] * t))
            

    return h

def complex_valued_ABCD(model):
    nr = model.Rr.shape[2]
    nc = model.Rc.shape[2]

    n_bar = nr + 2 * nc
    q_bar = model.R0.shape[0]
    m_bar = model.R0.shape[1]

    R_n = []
    p_n = []

    A_n = []
    B_n = []
    C_n = []

    for n in range(nr):
        # IMPLEMENT ME
        pass

    for n in range(nc):
        pass

    for n in range(nc):
        R_n.append(model.Rc[:, :, n])
        R_n.append(np.conj(model.Rc[:, :, n]))
        p_n.append(model.poles_complex[n])
        p_n.append(np.conj(model.poles_complex[n]))



    for n in range(n_bar):
        U, S, Vh = svd(R_n[n])
        rank =  S.shape[0]
        #A_n.append(np.block([[np.real(p_n[n]) * np.eye(rank), np.imag(p_n[n]*np.eye(rank))], [-np.imag(p_n[n]*np.eye(rank)), np.real(p_n[n]) * np.eye(rank)]]))
        A_n.append(p_n[n]*np.eye(rank))
        B_n.append(Vh.T)
        C_n.append((U@S).reshape(q_bar, rank))

    # A = block_diag(*A_n)
    poles_s = np.concatenate([model.poles_real, model.poles_complex, np.conj(model.poles_complex)])
    z, poles_z, k = bilinear_zpk([], poles_s, 0, 100)
    A = np.diag(poles_s)
    B = (np.block(B_n))
    B = B.T

    C = np.block(C_n)
    df = pd.DataFrame(A)
    df.to_csv("A_matrix.csv")

    D = model.R0
    return A, B, C, D

def real_valued_ABCD(model):
    nr = model.Rr.shape[2]
    nc = model.Rc.shape[2]

    n_bar = nr + 2 * nc
    q_bar = model.R0.shape[0]
    m_bar = model.R0.shape[1]

    A_n = []
    B_n = []
    C_n = []

    for n in range(len(model.poles_real)):
        U, S, Vh = svd(model.Rr[:, :, n])
        rank =  S.shape[0]
        #A_n.append(np.block([[np.real(p_n[n]) * np.eye(rank), np.imag(p_n[n]*np.eye(rank))], [-np.imag(p_n[n]*np.eye(rank)), np.real(p_n[n]) * np.eye(rank)]]))
        A_n.append(model.poles_real[n]*np.eye(rank))
        B_n.append(Vh.T)
        C_n.append((U@S).reshape(q_bar, rank))
    
    for n in range(len(model.poles_complex)):
            #R = R_n[n]
            U, S, Vh = svd(model.Rc[:, :, n])
            rank =  S.shape[0]
            A_n.append(np.block([[np.real(model.poles_complex[n]) * np.eye(rank), np.imag(model.poles_complex[n])*np.eye(rank)], [-np.imag(model.poles_complex[n])*np.eye(rank), np.real(model.poles_complex[n]) * np.eye(rank)]]))
            B_n.append(2*np.real(Vh))
            B_n.append(2*np.imag(Vh))
            C_n.append(np.real((U@S).reshape(q_bar, rank)))
            C_n.append(np.imag((U@S).reshape(q_bar, rank)))
    


    A = block_diag(*A_n)
    A = block_diag(*A_n)
    #A = np.diag(poles_z)
    B = (np.block(B_n))
    B = B.T
    C = np.block(C_n)
    D = model.R0

    df = pd.DataFrame(A)
    df.to_csv("A_matrix.csv")
    df = pd.DataFrame(B)
    df.to_csv("B_matrix.csv")
    df = pd.DataFrame(C)
    df.to_csv("C_matrix.csv")
    df = pd.DataFrame(D)
    df.to_csv("D_matrix.csv")
    return A, B, C, D


def state_space_ABCD(model, real_valued):
    if (real_valued):
        return real_valued_ABCD(model)
    else:
        return complex_valued_ABCD(model)


def state_space(model, dt):
    A, B, C, D = state_space_ABCD(model, True)
    sys = StateSpace(A, B, C, np.real(D))
    return sys

def FastVF(omega, H, order, options):
    # H = np.squeeze(np.asarray(H))
    if options == None:
        options = VF_Options()

    kbar = len(omega)
    qbar = np.shape(H)[1]
    mbar = np.shape(H)[2]
    nbar = order
    alpha = options.alpha

    # Ensure omega is a column vector
    omega = np.array(omega).reshape((kbar, 1))
    model = Pole_Residue_Model()

    num_real_poles = nbar % 2
    num_complex_pairs = (nbar - nbar % 2) // 2

    poles_real = np.array((num_real_poles))
    if num_real_poles == 1:
        poles_real = [-alpha*max(omega)]
    
    poles_complex = np.array((num_complex_pairs), dtype=complex)
    if num_complex_pairs == 1:
        poles_complex = (-alpha+1j)*np.max(omega) / 2
    elif np.min(omega) == 0:
        poles_complex = (-alpha+1j)*max(omega)*np.arange(1, num_complex_pairs + 1, dtype=complex)/num_complex_pairs
        #poles_complex = poles_complex.reshape((len(poles_complex), 1))
    else:
        poles_complex = (-alpha+1j)*(min(omega) + (max(omega)-min(omega))/(num_complex_pairs-1)*np.arange(0, num_complex_pairs, dtype=complex))
        #poles_complex = poles_complex.reshape((len(poles_complex), 1))
    
    iter = 1
    while iter <= options.max_iterations:
        phi_real, phi_complex = compute_phi_matrices(omega, poles_real, poles_complex)

        M = np.zeros((2*kbar, 2*nbar+1), dtype=complex)
        A = np.zeros((nbar*qbar*mbar,nbar), dtype=complex)
        b = np.zeros((nbar*qbar*mbar,1), dtype=complex)

        # Compute the first columns of M, which do not depend on q and m
        M[:kbar,0] = np.ones((kbar))
        M[:kbar,1:num_real_poles + 1] = np.real(phi_real)
        M[:kbar,num_real_poles+1:nbar+1] = np.real(phi_complex)
        M[kbar:,1:num_real_poles+1] = np.imag(phi_real)
        M[kbar:,num_real_poles+1:nbar+1] = np.imag(phi_complex)

        irow = 0
        for q in range(qbar):
            for m in range(mbar):
                V_H = np.squeeze(H[:, q, m])
                D_Hqm = diags(V_H, 0, shape=(kbar, kbar), format='csr')
                M[:kbar,nbar+1:nbar+num_real_poles+1] = -np.real(D_Hqm@phi_real) 
                M[:kbar,nbar+num_real_poles+1:] = -np.real(D_Hqm@phi_complex)
                M[kbar:,nbar+1:nbar+num_real_poles+1] = -np.imag(D_Hqm@phi_real)
                M[kbar:,nbar+num_real_poles+1:] = -np.imag(D_Hqm@phi_complex)

                # QR decomposition
                Q,R = np.linalg.qr(M,mode='reduced') 
                A[irow:irow+nbar,:] = R[nbar+1:,nbar+1:]
                b[irow:irow+nbar, 0] = Q[:kbar,nbar+1:].T @ np.real(V_H) + Q[kbar:,nbar+1:].T @ np.imag(V_H)
        
        # cw = Alsq\blsq;
        #cw = np.linalg.solve(A, b)
        cw = np.linalg.lstsq(A, b)[0]
        w = 1 + phi_real@cw[:num_real_poles] + phi_complex@cw[num_real_poles:] # DOUBLE CHECKED TO THIS POINT
        # Compute the new poles estimate
        A = np.zeros((nbar,nbar))
        bw = np.ones((nbar,1)) # DOUBLE CHECKED TO THIS POINT
        for i in range(num_real_poles):
            A[i,i] = poles_real[i]
            # bw(ii) = 1;

        for i in range(num_complex_pairs):
            A[num_real_poles+2*(i+1) - 2:num_real_poles+2*(i+1),num_real_poles+2*(i+1) - 2:num_real_poles+2*(i+1)] = np.array([
                    [np.real(poles_complex[i]),  np.imag(poles_complex[i])], 
                    [-np.imag(poles_complex[i]), np.real(poles_complex[i])]
                ])
            bw[num_real_poles+2*(i+1)-2:num_real_poles+2*(i+1),0] = np.array([2, 0])
            pass
            #A(nr+2*ii-1:nr+2*ii,nr+2*ii-1:nr+2*ii) = [real(pc(ii)), imag(pc(ii)); -imag(pc(ii)),real(pc(ii))]; 

        new_poles, _ = np.linalg.eig(A-bw@cw.conj().T)


        # Extract Real Poles
        eps = np.finfo(float).eps
        ind_rp = np.nonzero(np.abs(np.imag(new_poles)) < 10 * eps * np.abs(new_poles))[0] # I increased the constant 10 to 100 for now

        poles_real = np.real(new_poles[ind_rp])
        # num_real_poles = len(ind_rp)
        
        # Extract complex conjugate pairs of poles
        # Find only the poles with positive imaginary part
        ind_cp = np.nonzero(np.imag(new_poles)>=10*eps*abs(new_poles))[0]
        # ind_cp = np.nonzero(np.imag(new_poles)>=100*eps*abs(new_poles))[0]
        poles_complex = new_poles[ind_cp]
        num_complex_pairs = len(ind_cp)

        num_real_poles = len(poles_real)


        while len(poles_real) + 2*len(poles_complex) < order:
            mask = np.argmin(np.abs(np.imag(new_poles)))
            poles_real = np.append(poles_real, new_poles[mask])

        num_real_poles = len(poles_real)

        while 2 * len(poles_complex) > order:
            most_real_index = np.argmin(np.abs(np.imag(poles_complex)))
            poles_real = np.append(poles_real, poles_complex[most_real_index])
            poles_complex = np.delete(poles_complex, most_real_index)
        
        num_complex_pairs = len(poles_complex)
        
        while len(poles_real) + 2*len(poles_complex) > order:
            if len(poles_real) == 1:
                poles_real = np.array([])
                break
            magnitudes = np.abs(poles_real)

            # Step 2: Calculate the differences in magnitude between neighboring elements
            # Calculate the differences between magnitudes of consecutive elements
            diffs = np.abs(np.diff(magnitudes))

            # Step 3: Find the index of the element with the smallest difference in magnitude with its neighbor
            # Since we calculate differences between consecutive elements, the index of the smallest difference
            # will be between the i-th and (i+1)-th elements.
            # We take the first element of the pair to remove.
            index_of_min_diff = np.argmin(diffs)

            # Since we want to remove the element with the smallest difference, we choose the one with the
            # smaller magnitude if we're between a pair. We choose the min of the index of min difference and the next element.
            if index_of_min_diff == len(magnitudes) - 1:
                index_to_remove = index_of_min_diff  # If it's the last element, we can only remove this one
            else:
                if magnitudes[index_of_min_diff] <= magnitudes[index_of_min_diff + 1]:
                    index_to_remove = index_of_min_diff
                else:
                    index_to_remove = index_of_min_diff + 1

            # Step 4: Delete the element at the found index
            poles_real = np.delete(poles_real, index_to_remove)

        
        num_real_poles = len(poles_real)
        #poles_real = np.real(poles_real)

        # Stability/causality enforcement
        if options.enforce_stability:
            poles_real = -np.abs(poles_real)
            poles_complex = -np.abs(np.real(poles_complex)) + 1j*np.imag(poles_complex)

            # mask_real = np.abs(poles_real) > 1
            # mask_complex = np.abs(poles_complex) > 1
            # poles_real[mask_real] = 1 / poles_real[mask_real]
            # poles_complex[mask_complex] = 1 / poles_complex[mask_complex]

            # poles_real[mask_real] = poles_real[mask_real] / np.abs(poles_real[mask_real])
            # poles_complex[mask_complex] = poles_complex[mask_complex] / np.abs(poles_complex[mask_complex])

        if options.debug:
            plt.scatter(np.real(poles_complex), np.imag(poles_complex))
            plt.scatter(np.real(poles_real), np.imag(poles_real))
            plt.xlabel('Real')
            plt.ylabel('Imaginary')
            plt.title(f"Poles estimate, iteration {i}")
        
        
        # First convergence test
    
        w_minus_one = 1/np.sqrt(kbar)*np.linalg.norm(np.abs(w-1))
        # Do a tentative model fitting if either:
        # - the first convergence test is successful 
        # - Options.debug is enabled
        if w_minus_one <= options.poles_estimation_threshold:
            print(f'Convergence test (poles estimation): \t\tpassed ({w_minus_one})\n')
        else:
            print(f'Convergence test (poles estimation): \t\tfailed ({w_minus_one})\n')
        
        if  w_minus_one <= options.poles_estimation_threshold or options.debug:
            # Tentative final fitting
            phi_real, phi_complex = compute_phi_matrices(omega,poles_real,poles_complex)
            
            # Compute the matrix of the least squares problem
            A = np.zeros((2*kbar,nbar+1))
            A[:kbar,0] = np.ones((kbar))
            A[:kbar,1:num_real_poles+1] = np.real(phi_real)
            A[:kbar,num_real_poles+1:nbar+1] = np.real(phi_complex)
            A[kbar:,1:num_real_poles+1] = np.imag(phi_real)
            A[kbar:,num_real_poles+1:nbar+1] = np.imag(phi_complex) 
            
            # Store model coefficients in the output structure Model, in case it
            # will found accurate enough
            model.poles_real = poles_real
            model.poles_complex = poles_complex
            model.R0 = np.zeros((qbar,mbar), dtype=complex)
            model.Rr = np.zeros((qbar,mbar,num_real_poles), dtype=complex)
            model.Rc = np.zeros((qbar,mbar,num_complex_pairs), dtype=complex)

            # Model-samples error
            err = 0
            for q in range(qbar):
                for m in range(mbar):
                    # Right hand side
                    V_H = np.squeeze(H[:, q, m])
                    b = np.block([np.real(V_H), np.imag(V_H)])
                    c_H, _, _, _ = np.linalg.lstsq(A, b)
                    model.R0[q,m] = c_H[0]
                    model.Rr[q,m,:] = c_H[1:num_real_poles+1]
                    real_indices = slice(num_real_poles+1, None, 2)  # Equivalent to nr+2:2:end in MATLAB
                    imag_indices = slice(num_real_poles+2, None, 2)  # Equivalent to nr+3:2:end in MATLAB
                    model.Rc[q,m,:] = c_H[real_indices] + 1j*c_H[imag_indices]
                    
                    # Plot the given samples vs the model response for the
                    # (1,1) entry of the transfer function (if in debug mode)
                    if options.debug and q == 1 and m == 1:
                        # Compute model response
                        pass
            #             Htemp = compute_model_response(omega,model.R0(q,m),model.Rr(q,m,:),Model.Rc(q,m,:),Model.pr,Model.pc)
            #                 plot(omega,abs(squeeze(H(q,m,:))),'bx')
            #                 plot(omega,abs(squeeze(Htemp(q,m,:))),'r-.')
            #                 xlabel('Omega')
            #                 ylabel('Magnitude')
            #                 legend('Samples H_k','Model')

            #                 plot(omega,180/pi*angle(squeeze(H(q,m,:))),'bx')
            #                 plot(omega,180/pi*angle(squeeze(Htemp(q,m,:))),'r-.')
            #                 xlabel('Omega')
            #                 ylabel('Phase [deg]')
            #                 legend('Samples H_k','Model')
                    
                    err = err+np.sum(abs(A@c_H-b)**2)
                    pass

            err = np.sqrt(err)/np.sqrt(qbar*mbar*kbar)
            if err <= options.model_error_threshold:
                print('Convergence test (model-samples error): \tpassed (%e)\n',err)
                print('Model identification successful\n')
                # print('Modeling time: %f s\n',toc)
                return model
            else:
                print('Convergence test (model-samples error): \tfailed (%e)\n',err);
        
        iter += 1

    print('Warning: could not reach the desired modeling error within the allowed number of iterations\n')
    return model

#### OLD ##### 
#     for n in range(nr):
#         # IMPLEMENT ME
#         pass

#     for n in range(nc):
#         pass

#     for n in range(nc):
#         pass
#     #     R_n.append(model.Rc[:, :, n])
#     #     R_n.append(np.conj(model.Rc[:, :, n]))
#     #     p_n.append(model.poles_complex[n])
#     #     p_n.append(np.conj(model.poles_complex[n]))



#     # for n in range(n_bar):
#     #     #R = R_n[n]
#     #     U, S, Vh = svd(R_n[n])
#     #     rank =  S.shape[0]
#     #     A_n.append(np.block([[np.real(p_n[n]) * np.eye(rank), np.imag(p_n[n]*np.eye(rank))], [-np.imag(p_n[n]*np.eye(rank)), np.real(p_n[n]) * np.eye(rank)]]))

#     A = block_diag(*A_n)
#     df = pd.DataFrame(A)
#     df.to_csv("A_matrix.csv")

#     D = model.R0
#     return A, B, C, D
