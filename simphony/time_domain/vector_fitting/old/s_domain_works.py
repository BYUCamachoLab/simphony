import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.constants import speed_of_light

from simphony.simulation.jax_tools import python_based_while_loop

import matplotlib.pyplot as plt

from time import time

# @jax.jit
def _initial_poles(model_order, frequency, alpha):
    f = 1*jnp.linspace(jnp.min(frequency), jnp.max(frequency), model_order//2)
    poles = (-0.1 + 1j)*(2*jnp.pi*f)
    poles = jnp.concatenate([poles, poles.conj()])
    return poles

# @jax.jit
def _phi_matrices(frequency, poles):
    s = 2j*jnp.pi * frequency
    phi1 = 1 / (s[:, None] - poles[None, :])

    unity_column = jnp.ones((len(s), 1))
    
    phi0 = jnp.hstack((unity_column, phi1))

    return phi0, phi1

def _lstsq_matrices(model_order, transfer_function, phi0, phi1):
    """
    Here we perform the modified gram schmidt orthonalization on the block matrix [A1, A2] described here:
    https://arxiv.org/pdf/2208.06194
    This allows us to implement the Fast Vector Fitting algorithm:
    https://scholar.googleusercontent.com/scholar?q=cache:u4aY-dn1tF8J:scholar.google.com/+piero+triverio+vector+fitting&hl=en&as_sdt=0,45

    """
    num_ports = transfer_function.shape[1]
    M = jnp.zeros(((num_ports**2) * (model_order), (model_order)), dtype=complex)
    B = jnp.zeros(((num_ports**2) * (model_order)), dtype=complex)
    
    A1 = phi0
    Q1, R11 = jnp.linalg.qr(A1)
    
    iter = 0
    for i in range(num_ports):
        for j in range(num_ports):
            D = jnp.diag(transfer_function[:, i, j])
            A_block = jnp.hstack([phi0, -D @ phi1])            # never build the big matrix
            Q, R = jnp.linalg.qr(A_block, mode='reduced')
            
            R11 = R[:model_order+1, :model_order+1]
            R12 = R[:model_order+1, model_order+1:]
            R22 = R[model_order+1:, model_order+1:]
            Q2 = Q[:, model_order+1:]

            V = transfer_function[:, i, j]
            M = M.at[(iter) * (model_order) : (iter+1) * (model_order), :].set(R22)
            B = B.at[(iter) * (model_order) : (iter+1) * (model_order)].set(Q2.conj().T @ V)
            iter += 1
            
            
            
            # D = jnp.diag(transfer_function[:, i, j])
            # A2 = -D @ phi1
            
            # R12 = Q1.conj().T @ A2
            # Q2, R22 = jnp.linalg.qr(A2 - Q1 @ R12)

            # lhs = jnp.block([phi0, -D@phi1])

            # V = transfer_function[:, i, j]
            # M = M.at[(iter) * model_order : (iter + 1) * model_order, :].set(R22)
            # B = B.at[(iter) * model_order : (iter + 1) * model_order].set(Q2.conj().T @ V)
            # iter += 1

    return M, B

import numpy as np

def _full_lstsq_matrices(transfer_function, phi0, phi1):
        D = []
        V = []
        for i in range(transfer_function.shape[1]):
            for j in range(transfer_function.shape[2]):
                D.append(jnp.diag(transfer_function[:, i, j]))
                V.append(transfer_function[:, i, j])

        phi_column = []
        for _D in D:
            phi_column.append(-_D@phi1)

        blocks = [phi0] * (transfer_function.shape[1]*transfer_function.shape[2])
        M = jax.scipy.linalg.block_diag(*blocks)

        phi_column = np.vstack(phi_column)
        M = np.hstack([M, phi_column])

        return M, np.hstack(V)

# def _full_lstsq_matrices(tf, phi0, phi1):
#     """
#     Build the full least-squares matrices M and B from Phi0, Phi1, and a 3D transfer function array.

#     Parameters
#     ----------
#     phi0 : ndarray, shape (num_measurements, n0)
#         Phi0^(i) block.
#     phi1 : ndarray, shape (num_measurements, n1)
#         Phi1^(i) block.
#     tf : ndarray, shape (num_measurements, q, m)
#         Complex transfer function.
#         tf[:, j, k] = transfer function for output port j, input port k,
#         as a vector over 'num_measurements'.

#     Returns
#     -------
#     M : ndarray
#         Full LHS matrix in the least squares problem (complex).
#     B : ndarray
#         Full RHS stacked vector (complex).
#     """
#     num_meas, q, m_ports = tf.shape
#     m_rows, n0 = phi0.shape
#     _, n1 = phi1.shape

#     if num_meas != m_rows:
#         raise ValueError("phi0/phi1 first dimension must match num_measurements in tf.")

#     total_cols = q * n0 + n1
#     total_rows = q * num_meas

#     M = np.zeros((total_rows, total_cols), dtype=complex)
#     B = np.zeros((total_rows, 1), dtype=complex)

#     for j in range(q):
#         row_start = j * num_meas
#         row_end = row_start + num_meas

#         # Fill diagonal Phi0 block for c_H_j
#         col_start_H = j * n0
#         col_end_H = col_start_H + n0
#         M[row_start:row_end, col_start_H:col_end_H] = phi0

#         # Transfer function vector for this block row:
#         # For now, we'll assume we're fitting one specific input port k (or aggregated),
#         # so take one column from tf for fixed k. If you want multiple, adapt this.
#         tf_vec = tf[:, j, 0]  # <-- pick input port index here if needed

#         # Create D from this transfer function vector
#         D = np.diag(tf_vec)

#         # Fill last block column for c_w
#         col_start_w = q * n0
#         M[row_start:row_end, col_start_w:] = -D @ phi1

#         # Fill RHS vector
#         B[row_start:row_end, 0] = tf_vec

#     return M, B


# def _lstsq_matrices(model_order, transfer_function, phi0, phi1):
#     """
#     Real-stacked QR like in Triverio:
#       A_block = [[ Re(phi0) , -Re(D phi1) ],
#                  [ Im(phi0) , -Im(D phi1) ]]
#       Do QR(A_block), keep R22 and Q2, and build the stacked M and B.

#     transfer_function: (K, q, m) complex
#     Returns:
#       M: ((q*m)*model_order, model_order)
#       B: ((q*m)*model_order,)
#     """
#     K, q, m = transfer_function.shape
#     num_pairs = q * m
#     M = jnp.zeros((num_pairs * model_order, model_order))
#     B = jnp.zeros((num_pairs * model_order, ), dtype=transfer_function.dtype)

#     # reshape ports into one axis of length num_pairs
#     TF = transfer_function.reshape(K, -1)   # K x (q*m)

#     def process_one(V):
#         # D @ phi1 == phi1 * V[:, None] (avoid building a KxK diag)
#         Dphi1 = phi1 * V[:, None]
#         # build real-stacked block matrix
#         A_top = jnp.hstack([jnp.real(phi0), -jnp.real(Dphi1)])
#         A_bot = jnp.hstack([jnp.imag(phi0), -jnp.imag(Dphi1)])
#         A_block = jnp.vstack([A_top, A_bot])                   # (2K) x (N+1+N)

#         Q, R = jnp.linalg.qr(A_block, mode="reduced")
#         # partition
#         R22 = R[(model_order+1):, (model_order+1):]            # N x N
#         Q2  = Q[:, (model_order+1):]                           # (2K) x N

#         b = jnp.concatenate([jnp.real(V), jnp.imag(V)])        # (2K,)
#         rhs = Q2.T @ b                                         # (N,)
#         return R22, rhs

#     R_blocks, rhs_blocks = jax.vmap(process_one, in_axes=1)(TF)  # over port-pairs axis

#     # stack into big M and B
#     M = R_blocks.reshape((-1, model_order))
#     B = rhs_blocks.reshape((-1,))
#     return M, B

# def _lstsq_matrices(model_order, transfer_function, phi0, phi1):
#     num_ports = transfer_function.shape[1]

#     # Precompute QR of A1 since it’s constant
#     Q1, R11 = jnp.linalg.qr(phi0)

#     def process_pair(V):
#         # Avoid jnp.diag by elementwise multiply
#         A2 = -phi1 * V[:, None]
#         R12 = Q1.conj().T @ A2
#         Q2, R22 = jnp.linalg.qr(A2 - Q1 @ R12)
#         b_row = Q2.conj().T @ V
#         return R22, b_row

#     # Flatten all (i,j) pairs into a single axis
#     transfer_pairs = transfer_function.reshape(transfer_function.shape[0], -1)
#     R_blocks, b_blocks = jax.vmap(process_pair, in_axes=1)(transfer_pairs)

#     # Stack results
#     M = R_blocks.reshape((-1, model_order))
#     B = b_blocks.reshape((-1,))

#     return M, B

def _weight_error(frequency, poles_prev, weight_coeffs):
    s = 2j*jnp.pi * frequency
    terms = weight_coeffs / (s[:, None] - poles_prev)
    weights = 1.0 + jnp.sum(terms, axis=1)
    return jnp.sqrt(1/weights.shape[0] * jnp.sum(jnp.abs(weights - 1)**2))

# @jax.jit
def _fit_to_poles(transfer_function, frequency, poles):
    model_order = poles.shape[0]
    num_ports = transfer_function.shape[1]
    phi0, _ = _phi_matrices(frequency, poles)
    transfer_pairs = transfer_function.reshape(transfer_function.shape[0], -1)  # shape: (num_freq, num_ports*num_ports)

    # Define a function to solve lstsq for one port pair vector V (shape num_freq,)
    
    def solve_lstsq(V):
        sol, *_ = jnp.linalg.lstsq(phi0, V, rcond=None)
        return sol  # shape (model_order + 1,)

    # Vectorize over all port pairs (along axis=1)
    solutions = jax.vmap(solve_lstsq, in_axes=1)(transfer_pairs)  # shape (num_ports*num_ports, model_order + 1)

    # Reshape solutions back to (num_ports, num_ports, model_order + 1)
    solutions = solutions.reshape((num_ports, num_ports, model_order + 1))

    # Extract feedthrough (constant term)
    feedthrough = solutions[:, :, 0]  # shape (num_ports, num_ports)

    # Extract residues (remaining terms)
    residues = solutions[:, :, 1:].transpose(2, 0, 1)  # shape (model_order, num_ports, num_ports)

    return residues, feedthrough

# @jax.jit
def pole_residue_response(frequency, poles, residues, feedthrough):
    s = 2j*jnp.pi * (frequency)
    frequency_response = feedthrough[None, :, :] + jnp.sum(
    residues[None, :, :, :] / (s[:, None, None, None] - poles[None, :, None, None]),
    axis=1
)
    return frequency_response

# @jax.jit
def _mean_squared_error(transfer_function, frequency, poles, residues, feedthrough):
    fit = pole_residue_response(frequency, poles, residues, feedthrough)
    error = jnp.mean(jnp.abs(transfer_function - fit) ** 2)

    return error

def _enforce_conjugacy(poles):
    """
    Force conjugate symmetry by pairing poles with +/- imag parts.
    Keeps real poles as real.
    """
    # sort by imag part
    idx = jnp.argsort(jnp.imag(poles))
    p_sorted = poles[idx]

    # average small imag parts to pure real
    eps = 1e-12
    is_real = jnp.isclose(jnp.imag(p_sorted), 0.0, atol=1e-10)
    p_sorted = jnp.where(is_real, jnp.real(p_sorted) + 0j, p_sorted)

    # enforce conjugate pairs: take positive imag half and mirror them
    pos_mask = jnp.imag(p_sorted) > eps
    p_pos = p_sorted[pos_mask]
    p_neg = jnp.conj(p_pos)
    p_real = p_sorted[~pos_mask & ~(~pos_mask & (jnp.imag(p_sorted) < -eps))]  # approximately real left over

    # rebuild: interleave for stability (pos, conj), then reattach reals
    rebuilt = jnp.concatenate([jnp.ravel(jnp.column_stack([p_pos, p_neg])) if p_pos.size else jnp.array([], poles.dtype),
                               p_real])
    # if sizes mismatch due to odd order, pad/truncate to original length
    if rebuilt.size < poles.size:
        pad = jnp.zeros((poles.size - rebuilt.size,), dtype=poles.dtype)
        rebuilt = jnp.concatenate([rebuilt, pad + (-1.0 + 0j)])
    elif rebuilt.size > poles.size:
        rebuilt = rebuilt[:poles.size]
    return rebuilt

def vector_fitting(
    model_order,  
    transfer_function, 
    frequency,
    max_iterations=10,
    alpha=0.01,
    weight_threshold=0.0,
):
    
    # baseband_frequency = frequency 
    
    def poles_not_converged(state):
        _, weight_error, iteration = state

        return (iteration < max_iterations) & (weight_error > weight_threshold)

    def relocate_poles(state):
        previous_poles, _, iteration = state
        phi0, phi1 = _phi_matrices(frequency, previous_poles)
        M, B = _lstsq_matrices(model_order, transfer_function, phi0, phi1)
        weight_coeffs, *_ = jnp.linalg.lstsq(M, B)
        # M_prime, B_prime = _full_lstsq_matrices(transfer_function, phi0, phi1)
        # coeffs, *_ = jnp.linalg.lstsq(M_prime, B_prime)
        # weight_coeffs = coeffs[transfer_function.shape[1]*transfer_function.shape[2]*(model_order+1):]
        weight_error = _weight_error(frequency, previous_poles, weight_coeffs)

        A = jnp.diag(previous_poles)
        current_poles, _ = jnp.linalg.eig(A - jnp.ones((model_order, 1))@weight_coeffs[:, None].T)
        mask = current_poles.real > 0
        current_poles = jnp.where(mask, -current_poles.real+1j*current_poles.imag, current_poles)
        # current_poles = _enforce_conjugacy(current_poles)

        return (current_poles, weight_error, iteration + 1)

    initial_poles = _initial_poles(model_order, frequency, alpha)
    initial_state = (initial_poles, jnp.inf, 0)
    # final_poles, *_ = jax.lax.while_loop(poles_not_converged, relocate_poles, initial_state)
    final_poles, *_ = python_based_while_loop(poles_not_converged, relocate_poles, initial_state)
    # final_poles = initial_poles
    residues, feedthrough = _fit_to_poles(transfer_function, frequency, final_poles)
    error = _mean_squared_error(transfer_function, frequency, final_poles, residues, feedthrough)
    return final_poles, residues, feedthrough, error

def optimize_order(bias_fn, min_order, max_order):
    """
    bias_fn is a function of order which returns the MSE:
    https://ieeexplore.ieee.org/abstract/document/10274284?casa_token=rnFq1k0dt48AAAAA:nWbftIlFFN_x_a5oZ_CER3WTMeCXcAsvapSF8-SiLfi7seo-6rWv0TPWPLQkIaxEgtUr-w
    """ 
    C_min, *_ = bias_fn(min_order)
    C_max, *_ = bias_fn(max_order)
    C_max_minus_1, *_ = bias_fn(max_order-1)
    lambda_lower = jnp.abs(C_max_minus_1 - C_max)
    lambda_upper = C_min - C_max
    l = jnp.log10(lambda_lower)
    u = jnp.log10(lambda_upper)
    complexity_penalty = 10**(0.5*(u + l))

    # TODO: implement Golden Section Search
    # to minimize C - complexity_penalty * order
    golden_ratio = (jnp.sqrt(5) - 1) / 2
    a = min_order
    b = max_order
    c = int(b - golden_ratio * (b - a))
    d = int(a + golden_ratio * (b - a))

    fc = bias_fn(c)[0] + complexity_penalty*d
    fd = bias_fn(d)[0] + complexity_penalty*d
    while abs(b-a) > 1:
        if fc < fd:  # minimum is in [a, d]
            b, d, fd = d, c, fc
            c = int(b - golden_ratio * (b - a))
            fc = bias_fn(c)[0] + complexity_penalty*c
        else:        # minimum is in [c, b]
            a, c, fc = c, d, fd
            d = int(a + golden_ratio * (b - a))
            fd = bias_fn(d)[0] + complexity_penalty*d

    best_order = int(round((a + b) / 2))

    return bias_fn(best_order)


def vector_fitting_optimize_order(
    min_order,
    max_order,  
    transfer_function, 
    frequency,
    max_iterations=10,
    alpha=0.01,
    weight_threshold=0.0,
):
    def bias_fn(model_order):
        poles, residues, feedthrough, mean_squared_error = vector_fitting(
                                                                model_order, 
                                                                transfer_function, 
                                                                frequency,
                                                                max_iterations=max_iterations,
                                                                alpha=alpha,
                                                                weight_threshold=weight_threshold,
                                                            )
        return mean_squared_error, poles, residues, feedthrough

    mean_squared_error, poles, residues, feedthrough = optimize_order(bias_fn, min_order, max_order)

    return poles, residues, feedthrough, mean_squared_error


def main():
    from simphony.libraries import ideal, siepic
    from simphony.utils import dict_to_matrix
    import sax
    from time import time
    poles = jnp.array([-1.3578, -1.2679, -1.4851 + 0.2443*1j,-1.4851 - 0.2443*1j, -0.8487 + 2.9019*1j,-0.8487 - 2.9019*1j, -0.8587+3.1752*1j,-0.8587-3.1752*1j, -0.2497+6.5369*1j, -0.2497-6.5369*1j])
    residues = jnp.array([0.1059, -0.2808, 0.1166, 0.9569 - 0.7639*1j,0.9569 + 0.7639*1j, 0.9357 - 0.7593 * 1j,0.9357 + 0.7593 * 1j, 0.4579-0.7406*1j,0.4579+0.7406*1j, 0.2405-0.7437*1j, 0.2405+0.7437*1j])
    residues = jnp.reshape(residues[1:], (10, 1, 1))
    feedthrough = jnp.zeros((1, 1), dtype=complex)
    N = len(poles)

    f = jnp.linspace(0.001, 10/(2*jnp.pi), 100)
    aortic_response = pole_residue_response(f, poles, residues, feedthrough)
    
    
    _mzi, info = sax.circuit(
    netlist={
            "instances": {
                "gc_in": "gc",
                "splitter": "ybranch",
                "long_wg": "waveguide",
                "short_wg": "waveguide",
                "combiner": "ybranch",
                # "gc_out": "gc",
            },
            "connections": {
                "splitter,port 2": "long_wg,o0",
                "splitter,port 3": "short_wg,o0",
                "long_wg,o1": "combiner,port 2",
                "short_wg,o1": "combiner,port 3",
                # "combiner,port 1": "gc_out,o0",
            },
            "ports": {
                "in": "splitter,port 1",
                "out": "combiner,port 1",
            },
        },
        models={
            "ybranch": siepic.y_branch,
            "waveguide": siepic.waveguide,
            # "gc": siepic.grating_coupler,
        }
    )

    def mzi(wl=1.55):
        return _mzi(wl=wl, long_wg={"length": 10.0}, short_wg={"length": 40.0})

    f_min = speed_of_light / 1.6e-6
    f_max = speed_of_light / 1.50e-6
    # f_min = speed_of_light / 1.565e-6
    # f_max = speed_of_light / 1.5350e-6
    f_center = 0.5*(f_min+f_max)
    frequency = jnp.linspace(f_min, f_max, 1000)
    s_params = jnp.reshape(dict_to_matrix(mzi(wl=1e6*speed_of_light/frequency))[:, 0, 1], (1000, 1, 1))
    phase = jnp.unwrap(jnp.angle(s_params))
    s_params = s_params
    s_params = jnp.exp(-2j*phase)*s_params
    # s_params = s_params.at[:, 0, 0].set(0)
    # s_params = s_params.at[:, 1, 1].set(0)
    poles1, residues1, feedthrough1, error = vector_fitting(100, s_params, 1e-12*(frequency), max_iterations=15, alpha=0.01)

    # tic = time()
    
    # toc = time()
    # elapsed_time_1 = toc - tic
    # f_prime = jnp.linspace(f_min - 100e12, f_max+100e12, 1000)

    H = pole_residue_response(1e-12*(frequency), poles1, residues1, feedthrough1)
    plt.plot(frequency, jnp.angle(H[:, 0, 1]))
    plt.plot(frequency, jnp.angle(s_params[:, 0, 1]), 'r--')
    plt.show()
    pass

def main2():
    poles = jnp.array([-1.3578, -1.2679, -1.4851 + 0.2443*1j,-1.4851 - 0.2443*1j, -0.8487 + 2.9019*1j,-0.8487 - 2.9019*1j, -0.8587+3.1752*1j,-0.8587-3.1752*1j, -0.2497+6.5369*1j, -0.2497-6.5369*1j])
    residues = jnp.array([0.1059, -0.2808, 0.1166, 0.9569 - 0.7639*1j,0.9569 + 0.7639*1j, 0.9357 - 0.7593 * 1j,0.9357 + 0.7593 * 1j, 0.4579-0.7406*1j,0.4579+0.7406*1j, 0.2405-0.7437*1j, 0.2405+0.7437*1j])
    residues = jnp.reshape(residues[1:], (10, 1, 1))
    feedthrough = jnp.zeros((1, 1), dtype=complex)
    N = len(poles)

    f = jnp.linspace(0.001, 10/(2*jnp.pi), 100)
    response = pole_residue_response(f, poles, residues, feedthrough)

    poles1, residues1, feedthrough1, error = vector_fitting(10, response, f, max_iterations=5)
    toc = time()
    H = pole_residue_response(f, poles1, residues1, feedthrough1)
    plt.plot(f, jnp.abs(H[:, 0, 0])**2)
    plt.plot(f, jnp.abs(response[:, 0, 0])**2, 'r--')
    plt.show()

    plt.scatter(poles.real, poles.imag, marker='x')
    plt.scatter(poles1.real, poles1.imag, marker='o')
    plt.show()
    pass

if __name__ == "__main__":
    main()
    # main2()