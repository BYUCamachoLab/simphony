import jax
import jax.numpy as jnp
from scipy.constants import speed_of_light
from scipy.optimize import linear_sum_assignment

# from simphony.simulation.jax_tools import python_based_while_loop

import matplotlib.pyplot as plt

from time import time

from simphony.conventions import PHYSICIST, ENGINEER
from simphony.performance.performance import persistent_cache

# @jax.jit
def _initial_poles(model_order, frequency, sampling_frequency, gamma, sign_convention):
    f = jnp.linspace(jnp.min(frequency), jnp.max(frequency), model_order)
    poles = gamma*jnp.exp(sign_convention*1j*2*jnp.pi*f/sampling_frequency)
    return poles

# @jax.jit
def _phi_matrices(frequency, sampling_frequency, poles, sign_convention):
    z = jnp.exp(sign_convention*1j * 2 * jnp.pi * frequency / sampling_frequency)
    phi1 = 1 / (z[:, None] - poles[None, :])

    unity_column = jnp.ones((len(z), 1))
    
    phi0 = jnp.hstack((unity_column, phi1))

    return phi0, phi1

def _lstsq_matrices(model_order, transfer_function, phi0, phi1):
    """

    """
    # num_ports = transfer_function.shape[1]
    num_outputs = transfer_function.shape[1]
    num_inputs = transfer_function.shape[2]
    # M = jnp.zeros(((num_ports**2) * (model_order), (model_order)), dtype=complex)
    # B = jnp.zeros(((num_ports**2) * (model_order)), dtype=complex)
    M = jnp.zeros(((num_inputs*num_outputs) * (model_order), (model_order)), dtype=complex)
    B = jnp.zeros(((num_inputs*num_outputs) * (model_order)), dtype=complex)
    
    A1 = phi0
    Q1, R11 = jnp.linalg.qr(A1)
    
    iter = 0
    # for i in range(num_ports):
    #     for j in range(num_ports):
    for m in range(num_inputs):
        for q in range(num_outputs):
            D = jnp.diag(transfer_function[:, q, m])
            A_block = jnp.hstack([phi0, -D @ phi1])            # never build the big matrix
            Q, R = jnp.linalg.qr(A_block, mode='reduced')
            
            R11 = R[:model_order+1, :model_order+1]
            R12 = R[:model_order+1, model_order+1:]
            R22 = R[model_order+1:, model_order+1:]
            Q2 = Q[:, model_order+1:]

            V = transfer_function[:, q, m]
            M = M.at[(iter) * (model_order) : (iter+1) * (model_order), :].set(R22)
            B = B.at[(iter) * (model_order) : (iter+1) * (model_order)].set(Q2.conj().T @ V)
            iter += 1

    return M, B

def _weight_error(frequency, sampling_frequency, poles_prev, weight_coeffs, sign_convention):
    z = jnp.exp(sign_convention*1j * 2 * jnp.pi * frequency/sampling_frequency)
    terms = weight_coeffs / (z[:, None] - poles_prev)
    weights = 1.0 + jnp.sum(terms, axis=1)
    return jnp.sqrt(1/weights.shape[0] * jnp.sum(jnp.abs(weights - 1)**2))

# # @jax.jit
# def _fit_to_poles(transfer_function, frequency, sampling_frequency, poles):
#     model_order = poles.shape[0]
#     num_ports = transfer_function.shape[1]
#     residues = jnp.zeros((model_order, num_ports, num_ports), dtype=complex)
#     feedthrough = jnp.zeros((num_ports, num_ports), dtype=complex)
#     phi0, _ = _phi_matrices(frequency, sampling_frequency, poles)
#     for i in range(num_ports):
#         for j in range(num_ports):
#             # Q,R = np.linalg.qr(phi0,mode='reduced')
#             # solutions = np.linalg.pinv(R)@Q.conj().T@self.S[:, i, j]
#             solutions, *_ = jnp.linalg.lstsq(phi0, transfer_function[:, i, j], rcond=None)
#             feedthrough = feedthrough.at[i, j].set(jnp.array(solutions[0]))
#             residues = residues.at[:, i, j].set(solutions[1:])
    
#     return residues, feedthrough

# @jax.jit
def _fit_to_poles(transfer_function, frequency, sampling_frequency, poles, sign_convention):
    model_order = poles.shape[0]
    # num_ports = transfer_function.shape[1]
    num_outputs = transfer_function.shape[1]
    num_inputs = transfer_function.shape[2]
    phi0, _ = _phi_matrices(frequency, sampling_frequency, poles, sign_convention)
    # transfer_pairs = transfer_function.reshape(transfer_function.shape[0], -1)  # shape: (num_freq, num_ports*num_ports)
    transfer_pairs = transfer_function.transpose(1,2,0).reshape(-1, transfer_function.shape[0]).T
    # Define a function to solve lstsq for one port pair vector V (shape num_freq,)
    
    def solve_lstsq(V):
        sol, *_ = jnp.linalg.lstsq(phi0, V, rcond=None)
        return sol  # shape (model_order + 1,)

    # Vectorize over all port pairs (along axis=1)
    solutions = jax.vmap(solve_lstsq, in_axes=1)(transfer_pairs)  # shape (num_ports*num_ports, model_order + 1)

    # Reshape solutions back to (num_ports, num_ports, model_order + 1)
    solutions = solutions.reshape((num_outputs, num_inputs, model_order + 1))

    # Extract feedthrough (constant term)
    feedthrough = solutions[:, :, 0]  # shape (num_ports, num_ports)

    # Extract residues (remaining terms)
    residues = solutions[:, :, 1:].transpose(2, 0, 1)  # shape (model_order, num_ports, num_ports)

    return residues, feedthrough

# @jax.jit
def pole_residue_response_discrete(frequency, center_frequency, sampling_frequency, poles, residues, feedthrough, sign_convention=PHYSICIST):
    z = jnp.exp(sign_convention*1j * 2 * jnp.pi * (frequency-center_frequency)/sampling_frequency)
    frequency_response = feedthrough[None, :, :] + jnp.sum(
    residues[None, :, :, :] / (z[:, None, None, None] - poles[None, :, None, None]),
    axis=1
)
    return frequency_response

# @jax.jit
def _mean_squared_error(transfer_function, frequency, center_frequency, sampling_frequency, poles, residues, feedthrough, sign_convention):
    fit = pole_residue_response_discrete(frequency, center_frequency, sampling_frequency, poles, residues, feedthrough, sign_convention=PHYSICIST)
    error = jnp.mean(jnp.abs(transfer_function - fit) ** 2)

    return error

# def state_space_discrete(poles, residues, feedthrough):
#     model_order = poles.shape[0]
#     num_ports = feedthrough.shape[0]
#     A = jnp.zeros(
#         (model_order * num_ports, model_order * num_ports), dtype=complex
#     )
#     B = jnp.zeros((model_order * num_ports, num_ports), dtype=complex)
#     C = jnp.zeros((num_ports, model_order * num_ports), dtype=complex)
#     for i in range(model_order):
#         A = A.at[
#             i * num_ports : (i + 1) * num_ports,
#             i * num_ports : (i + 1) * num_ports,
#         ].set(poles[i] * jnp.eye(num_ports))
        
#         B = B.at[i * num_ports : (i + 1) * num_ports, :].set(jnp.eye(num_ports))
#         C = C.at[:, i * num_ports : (i + 1) * num_ports].set(residues[i, :, :])

#     return A, B, C, feedthrough

@persistent_cache
def vector_fitting_discrete(
    model_order,  
    transfer_function, 
    frequency,
    center_frequency,
    sampling_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    gamma=0.95,
    weight_threshold=0.0,
):
    # Convert to engineer's Sign Convention
    # transfer_function = jnp.conj(transfer_function)

    baseband_frequency = frequency - center_frequency
    
    def poles_not_converged(state):
        _, weight_error, iteration = state

        return (iteration < max_iterations) & (weight_error > weight_threshold)

    def relocate_poles(state):
        previous_poles, _, iteration = state
        phi0, phi1 = _phi_matrices(baseband_frequency, sampling_frequency, previous_poles, sign_convention)
        M, B = _lstsq_matrices(model_order, transfer_function, phi0, phi1)
        weight_coeffs, *_ = jnp.linalg.lstsq(M, B)

        weight_error = _weight_error(baseband_frequency, sampling_frequency, previous_poles, weight_coeffs, sign_convention)

        A = jnp.diag(previous_poles)
        current_poles, _ = jnp.linalg.eig(A - jnp.outer(jnp.ones(model_order), weight_coeffs))
        mask = jnp.abs(current_poles) > 1
        # current_poles = current_poles.at[mask].set(1 / current_poles[mask])
        current_poles = jnp.where(mask, 1 / current_poles, current_poles)

        return (current_poles, weight_error, iteration + 1)

    initial_poles = _initial_poles(model_order, baseband_frequency, sampling_frequency, gamma, sign_convention)
    initial_state = (initial_poles, jnp.inf, 0)
    final_poles, *_ = jax.lax.while_loop(poles_not_converged, relocate_poles, initial_state)
    residues, feedthrough = _fit_to_poles(transfer_function, baseband_frequency, sampling_frequency, final_poles, sign_convention)
    
    # Convert back to Physicist's Convention
    # final_poles = 1/final_poles
    # residues = -residues * final_poles[:, None, None]
    
    error = _mean_squared_error(transfer_function, frequency, center_frequency, sampling_frequency, final_poles, residues, feedthrough, sign_convention)
    
    
    return final_poles, residues, feedthrough, error



# @jax.jit
# def vector_fitting_z(
#     model_order,  
#     transfer_function, 
#     frequency,
#     center_frequency,
#     sampling_frequency,
#     max_iterations = 15,
#     gamma = 0.95,
# ):
#     baseband_frequency = frequency - center_frequency
#     poles = _initial_poles(model_order, baseband_frequency, sampling_frequency, gamma)
#     for _ in range(max_iterations):
#         phi0, phi1 = _phi_matrices(poles, baseband_frequency)
#         M, B = _lstsq_matrices(model_order, transfer_function, phi0, phi1)
#         weights, *_ = jnp.linalg.lstsq(M, B)
#         # weights_row = weights.reshape((len(weights), 1))
#         # unity_column = jnp.ones((model_order, 1))

#         A = jnp.diag(poles)
#         poles, _ = jnp.linalg.eig(A - jnp.outer(jnp.ones(model_order), weights))
#         mask = jnp.abs(poles) > 1
#         poles = poles.at[mask].set(1 / (poles[mask]))

#         error = compute_error()

#         if error < tolerable_error:
#             break


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


# TODO: Cache the model order, not the model itself to save space
@persistent_cache
def optimize_order_vector_fitting_discrete(
    min_order,
    max_order,  
    transfer_function, 
    frequency,
    center_frequency,
    sampling_frequency,
    sign_convention=PHYSICIST,
    max_iterations=10,
    gamma=0.95,
    weight_threshold=0.0,
):
    def bias_fn(model_order):
        poles, residues, feedthrough, mean_squared_error = vector_fitting_discrete(
                                                                model_order, 
                                                                transfer_function, 
                                                                frequency, 
                                                                center_frequency,
                                                                sampling_frequency,
                                                                sign_convention=sign_convention,
                                                                max_iterations=max_iterations,
                                                                gamma=gamma,
                                                                weight_threshold=weight_threshold,
                                                                # use_cache=False, ### TODO: Decide whether this be necessary
                                                            )
        return mean_squared_error, poles, residues, feedthrough


    mean_squared_error, poles, residues, feedthrough = optimize_order(bias_fn, min_order, max_order)

    return poles, residues, feedthrough, mean_squared_error


# def broken_state_space_discrete(poles, residues, feedthrough):
#     """
#     poles: array of shape (r,)
#     residues: array of shape (r, q, m)
#     D: feedthrough, shape (q, m)
    
#     Returns: A, B, C, D with replicated poles per input
#     """
#     r, q, m = residues.shape
#     M = r * m

#     # A: block-diagonal with replicated poles
#     A = jnp.repeat(jnp.diag(poles), m, axis=0)

#     # B: each input excites its replicated states
#     B = jnp.zeros((M, m), dtype=complex)
#     for i in range(r):
#         for j in range(m):
#             B = B.at[i*m + j, j].set(1.0)

#     # C: map states to outputs using residues
#     C = jnp.zeros((q, M), dtype=complex)
#     for i in range(r):
#         for j in range(m):
#             C = C.at[:, i*m + j].set(residues[i, :, j])

#     D = feedthrough
#     return A, B, C, D

def state_space_discrete(poles, residues, feedthrough):
    """
    Create a discrete-time state-space model without SVD.
    
    poles: shape (r,)
    residues: shape (r, q, m)
    feedthrough: shape (q, m)
    
    Returns A, B, C, D with replicated poles per input
    """
    r, q, m = residues.shape
    M = r * m  # total number of states

    # A: block-diagonal, replicate each pole m times
    A = jnp.kron(jnp.diag(poles), jnp.eye(m, dtype=complex))

    # B: each input excites its replicated states
    B = jnp.zeros((M, m), dtype=complex)
    for i in range(r):
        for j in range(m):
            B = B.at[i*m + j, j].set(1.0)

    # C: map states to outputs using residues
    C = jnp.zeros((q, M), dtype=complex)
    for i in range(r):      # over poles
        for j in range(m):  # over inputs
            # state index for this replicated pole
            idx = i*m + j
            # residues[i, :, j] has shape (q,)
            C = C.at[:, idx].set(residues[i, :, j])

    D = feedthrough
    return A, B, C, D



# def state_space_discrete(poles, residues, feedthrough):
#     """
#     Creates a state space model without the need for singular value decomposition

#     This approach is not guaranteed to produce a minimal model see 
#     Vector Fitting by Piero Triverio∗, August 27, 2019
#     """
#     # TODO: Make sure this works when num_inputs is not equal to num_outputs
#     model_order = poles.shape[0]
#     num_outputs = feedthrough.shape[0]
#     num_inputs = feedthrough.shape[1]
    
#     A = jnp.zeros(
#         (model_order * num_inputs, model_order * num_inputs), dtype=complex
#     )
#     B = jnp.zeros((model_order * num_inputs, num_inputs), dtype=complex)
#     C = jnp.zeros((num_outputs, model_order * num_inputs), dtype=complex)
#     for i in range(model_order):
#         A = A.at[i * num_inputs : (i + 1) * num_inputs, i * num_inputs : (i + 1) * num_inputs].set(poles[i] * jnp.eye(num_inputs))
#         B = B.at[i * num_inputs : (i + 1) * num_inputs, :].set(jnp.eye(num_inputs))
#         C = C.at[:, i * num_outputs : (i + 1) * self.state_space_matricesnum_inputs].set(residues[i, :, :])

#     D = feedthrough
#     return A, B, C, D

@jax.jit
def _state_space_response_discrete(A, B, C, D, u, x0):
    def step(x, u_k):
        y_k = C @ x + D @ u_k
        x_next = A @ x + B @ u_k
        return x_next, (y_k, x_next)

    _, (yout, xout) = jax.lax.scan(step, x0, u)
    return yout, xout


def state_space_response_discrete(A, B, C, D, u, x0=None):
    if x0 is None:
        x0 = jnp.zeros((A.shape[0],), dtype=A.dtype)

    return _state_space_response_discrete(A, B, C, D, u, x0)


@jax.jit
def _state_space_response_discrete_structured(A_diag, residues, D, u, x0, b_phase):
    r = residues.shape[0]
    m = residues.shape[2]

    def step(x, u_k):
        y_k = jnp.einsum('iqm,im->q', residues, x.reshape((r, m))) + D @ u_k
        x_next = A_diag * x + b_phase * jnp.tile(u_k, r)
        return x_next, (y_k, x_next)

    _, (yout, xout) = jax.lax.scan(step, x0, u)
    return yout, xout


def state_space_response_discrete_structured(A, B, C, D, phase, u, x0=None):
    """
    Fast discrete-time response for the structured vector-fitting realization.

    This assumes the ABCD matrices come from `state_space_discrete`, with:
    - A diagonal/block-diagonal replicated-pole structure.
    - B equal to the canonical replicated-input selector.
    - C ordered so it can be reshaped into residues with shape (r, q, m).
    - State ordering grouped by pole, then input.

    This is not equivalent to `state_space_response_discrete` for arbitrary
    state-space realizations for user-defined state space models. This method was
    explicitly designed to be used for the pole-residue models generated from user
    defined s-parameter matrices.
    """
    if x0 is None:
        x0 = jnp.zeros((A.shape[0],), dtype=A.dtype)

    M = A.shape[0]
    m = B.shape[1]
    r = M // m
    q = C.shape[0]

    A_diag = jnp.diag(A)
    residues = jnp.transpose(C.reshape(q, r, m), (1, 0, 2))

    return _state_space_response_discrete_structured(A_diag, residues, D, u, x0, phase)


# def state_space_response_discrete(A, B, C, D, u, x0=None):
#     # Initial state
#     if x0 is None:
#         x0 = jnp.zeros((A.shape[0],), dtype=A.dtype)

#     def step(x, u_k):
#         y_k = C @ x + D @ u_k
#         x_next = A @ x + B @ u_k
#         return x_next, (y_k, x)

#     # Run scan
#     x_final, (yout, xout) = jax.lax.scan(step, x0, u)

#     return yout, jnp.vstack([xout, x_final])

# def state_space_response_discrete(A, B, C, D, u, x=None):
#     out_samples = len(u)
#     # stoptime = (out_samples) * dt

#     xout = jnp.zeros((out_samples, A.shape[0]), dtype=complex)
#     yout = jnp.zeros((out_samples, C.shape[0]), dtype=complex)
#     # tout = jnp.linspace(0.0, stoptime, num=out_samples)

#     xout = xout.at[0, :].set(jnp.zeros((A.shape[1],), dtype=complex))

#     if x is not None:
#         xout = xout.at[0, :].set(x)

#     u_dt = u

#     # Simulate the system
#     for i in range(0, out_samples):
#         xout = xout.at[i+1, :].set(jnp.dot(A, xout[i, :]) + jnp.dot(B, u_dt[i, :]))
#         yout = yout.at[i, :].set(jnp.dot(C, xout[i, :]) + jnp.dot(D, u_dt[i, :]))

#     # Last point
#     yout = yout.at[out_samples - 1, :].set(jnp.dot(C, xout[out_samples - 1, :]) + jnp.dot(
#         D, u_dt[out_samples - 1, :]
#     ))

#     return yout, xout

def state_space_frequency_response_discrete(A, B, C, D, f, f_center, dt):
    """
    Compute the frequency response of a state-space system.

    Parameters
    ----------
    A : jnp.ndarray, shape (n, n)
        State matrix
    B : jnp.ndarray, shape (n, m)
        Input matrix
    C : jnp.ndarray, shape (q, n)
        Output matrix
    D : jnp.ndarray, shape (q, m)
        Feedthrough matrix
    freqs : jnp.ndarray
        Frequencies in Hz (or angular frequencies depending on jω convention)

    Returns
    -------
    H : jnp.ndarray, shape (q, m, len(freqs))
        Frequency response at each frequency
    """
    n_out, n_in = D.shape
    n_freq = f.size
    H = jnp.zeros((n_freq, n_out, n_in), dtype=complex)

    # Discrete-time z = e^(j*omega*dt)
    # TODO: Add in sign conventions
    z = jnp.exp(-1j * 2 * jnp.pi * (f-f_center) * dt)

    for k in range(n_freq):
        Hk = C @ jnp.linalg.inv(z[k] * jnp.eye(A.shape[0]) - A) @ B + D
        H = H.at[k, :, :].set(Hk)

    return H

    return H

def main():
    from simphony.libraries import ideal
    from simphony.utils import dict_to_matrix
    import sax
    from time import time

    netlist = {
        "instances": {
            "wg": "waveguide",
            "hr": "half_ring",
        },
        "connections": {
            "hr,o2": "wg,o0",
            "hr,o3": "wg,o1",
        },
        "ports": {
            "o0": "hr,o0",
            "o1": "hr,o1",
        }
    }

    circuit, info = sax.circuit(
        netlist=netlist,
        models={
            "waveguide": ideal.waveguide,
            "half_ring": ideal.coupler,
        }
    )

    f_min = speed_of_light / 1.6e-6
    f_max = speed_of_light / 1.5e-6
    f_center = 0.5*(f_min+f_max)
    # f_center = 192.9e12
    frequency = jnp.linspace(f_min, f_max, 1000)
    s_params = dict_to_matrix(circuit(wl=1e6*speed_of_light/frequency, wg={"length": 77.0, "loss": 100}))

    sampling_frequency = 1e14
    model_order = 10


    tic = time()
    poles, residues, feedthrough, error = optimize_order_vector_fitting_discrete(10, 50, s_params, frequency, f_center, sampling_frequency)
    toc = time()
    elapsed_time_1 = toc - tic
    model_order = len(poles)
    poles_eng, residues_eng, feedthrough_eng, erro = vector_fitting_discrete(model_order, jnp.conj(s_params), frequency, f_center, sampling_frequency, sign_convention=ENGINEER)
    # tic = time()
    # poles, residues, feedthrough, error = vector_fitting_z_optimize_order(10, 50, s_params, frequency, f_center, sampling_frequency)
    # toc = time()
    # elapsed_time_2 = toc - tic
    # tic = time()
    # poles, residues, feedthrough, error = vector_fitting_z_optimize_order(10, 50, s_params, frequency, f_center, sampling_frequency)
    # toc = time()
    # elapsed_time_3 = toc - tic
    # tic = time()
    # poles, residues, feedthrough, error = vector_fitting_z_optimize_order(10, 50, s_params, frequency, f_center, sampling_frequency)
    # toc = time()
    # elapsed_time_4 = toc - tic
    
    pass
    # tic = time()
    # poles, residues, feedthrough, error = vector_fitting_z(model_order, s_params, frequency, f_center, sampling_frequency)
    # toc = time()
    # elapsed_time = toc - tic
    # print(elapsed_time)
    # tic = time()
    # poles, residues, feedthrough, error = vector_fitting_z(model_order+1, s_params, frequency, f_center, sampling_frequency)
    # toc = time()
    # elapsed_time = toc - tic
    # print(elapsed_time)
    f = jnp.linspace(-sampling_frequency / 2, sampling_frequency / 2, 100000) + f_center

    plt.scatter(poles.real, poles.imag)
    plt.scatter(poles_eng.real, poles_eng.imag)
    plt.show()

    plt.scatter(residues[:, 0, 1].real, residues[:, 0, 1].imag)
    plt.scatter(residues_eng[:, 0, 1].real, residues_eng[:, 0, 1].imag)
    plt.show()


    H = pole_residue_response_discrete(f, f_center, sampling_frequency, poles, residues, feedthrough, sign_convention=PHYSICIST)
    H_eng = pole_residue_response_discrete(f, f_center, sampling_frequency, jnp.conj(poles), jnp.conj(residues), jnp.conj(feedthrough), sign_convention=ENGINEER)
    # plt.plot(f, jnp.abs(H[:, 0, 1]))
    plt.plot(f, jnp.abs(H_eng[:, 0, 1]), 'r--')
    # plt.plot(frequency, jnp.abs(s_params[:, 0, 1]))
    plt.plot(frequency, jnp.abs(s_params[:, 0, 1]))
    plt.show()

    # plt.plot(f, jnp.angle(H[:, 0, 1]))
    # plt.plot(frequency, jnp.angle(s_params[:, 0, 1]))
    plt.plot(f, jnp.angle(H_eng[:, 0, 1]), 'r--')
    plt.plot(frequency, jnp.angle(jnp.conj(s_params[:, 0, 1])))
    plt.xlim([187e12, 200e12])
    plt.show()

    t = jnp.arange(0, 5000/sampling_frequency, 1/sampling_frequency)
    A, B, C, D = state_space_discrete(poles, residues, feedthrough)
    u = jnp.zeros((t.shape[0], 2), dtype=complex)
    u = u.at[:, 0].set(1)
    u = u.at[:, 0].set(u[:, 0]*jnp.exp(-1j*2*jnp.pi*(194.2e12-f_center)*t))
    y, x = state_space_response_discrete(A, B, C, D, u)
    plt.plot(t, jnp.abs(y[:, 1])**2, label="shifted inputs", linewidth=3.0)
    # plt.show()

    u = u.at[:, 0].set(1)
    y, x = state_space_response_discrete(jnp.exp(1j*2*jnp.pi*(194.2e12 - f_center)/sampling_frequency)*A, jnp.exp(1j*2*jnp.pi*(194.2e12 - f_center)/sampling_frequency)*B, C, D, u)
    plt.plot(t, jnp.abs(y[:, 1])**2, 'r--', label="modified ss model")
    plt.xlabel("time (s)")
    plt.ylabel("mag squared")
    plt.legend()
    plt.show()
    # print(jnp.angle(y))
    # plt.plot(frequency, jnp.angle(s_params[:, 0, 1]))
    pass


def main2():
    sampling_frequency = 1
    center_frequency = sampling_frequency / 2
    frequency = jnp.linspace(-sampling_frequency/2, sampling_frequency/2, 1000) + center_frequency

    num_poles = 10
    key = jax.random.PRNGKey(42)
    pole_radius_key, pole_angle_key = jax.random.split(key)
    pole_radii = jax.random.uniform(pole_radius_key, (num_poles,), minval=0.10, maxval=0.98)
    pole_angles = jax.random.uniform(pole_angle_key, (num_poles,), minval=-jnp.pi, maxval=jnp.pi)
    poles = pole_radii * jnp.exp(1j * pole_angles)

    port_indices = jnp.arange(10, dtype=jnp.float32)
    row_indices = port_indices[:, None]
    col_indices = port_indices[None, :]
    base_residue = (
        0.9
        + 0.12 * row_indices
        - 0.05 * col_indices
        + 0.015 * row_indices * col_indices
        + 1j * (0.2 + 0.08 * row_indices + 0.045 * col_indices)
    )
    scale_magnitudes = jnp.linspace(1.00, 0.08, num_poles)
    scale_phases = jnp.linspace(0.0, 0.75 * jnp.pi, num_poles)
    residue_scales = scale_magnitudes * jnp.exp(1j * scale_phases)
    residues = residue_scales[:, None, None] * base_residue[None, :, :]

    feedthrough = jnp.zeros(residues.shape[1:], dtype=complex)

    def match_pole_residue_pairs(reference_poles, candidate_poles, candidate_residues):
        pole_distance = jnp.abs(reference_poles[:, None] - candidate_poles[None, :])
        row_indices, col_indices = linear_sum_assignment(pole_distance)
        assignment_order = jnp.asarray(col_indices[jnp.argsort(jnp.asarray(row_indices))])
        return candidate_poles[assignment_order], candidate_residues[assignment_order]

    original_poles = poles
    original_residues = residues
    original_feedthrough = feedthrough

    H = pole_residue_response_discrete(
        frequency,
        center_frequency,
        sampling_frequency,
        original_poles,
        original_residues,
        original_feedthrough,
    )
    final_poles, final_residues, final_feedthrough, fit_error = vector_fitting_discrete(
        model_order=num_poles,
        transfer_function=H,
        frequency=frequency,
        center_frequency=center_frequency,
        sampling_frequency=sampling_frequency,
    )

    matched_final_poles, matched_final_residues = match_pole_residue_pairs(
        original_poles,
        final_poles,
        final_residues,
    )
    fitted_response = pole_residue_response_discrete(
        frequency,
        center_frequency,
        sampling_frequency,
        final_poles,
        final_residues,
        final_feedthrough,
    )

    pole_error = jnp.max(jnp.abs(matched_final_poles - original_poles))
    residue_error = jnp.max(jnp.abs(matched_final_residues - original_residues))
    feedthrough_error = jnp.max(jnp.abs(final_feedthrough - original_feedthrough))
    response_error = jnp.max(jnp.abs(fitted_response - H))
    relative_response_error = jnp.linalg.norm(fitted_response - H) / jnp.maximum(
        jnp.linalg.norm(H),
        1e-12,
    )

    print("Fit MSE:", fit_error)
    print("Max pole error:", pole_error)
    print("Max residue error:", residue_error)
    print("Max feedthrough error:", feedthrough_error)
    print("Max response error:", response_error)
    print("Relative response error:", relative_response_error)

if __name__ == "__main__":
    main2()
