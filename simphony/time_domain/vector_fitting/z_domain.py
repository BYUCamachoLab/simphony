import jax
import jax.numpy as jnp
from scipy.constants import speed_of_light

# from simphony.simulation.jax_tools import python_based_while_loop

import matplotlib.pyplot as plt

from time import time

from simphony.conventions import PHYSICIST, ENGINEER

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

@jax.jit
def _fit_to_poles(transfer_function, frequency, sampling_frequency, poles, sign_convention):
    model_order = poles.shape[0]
    num_ports = transfer_function.shape[1]
    phi0, _ = _phi_matrices(frequency, sampling_frequency, poles, sign_convention)
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

@jax.jit
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

def vector_fitting_discrete(
    model_order,  
    transfer_function, 
    frequency,
    center_frequency,
    sampling_frequency,
    sign_convention=PHYSICIST,
    max_iterations=40,
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
                                                                weight_threshold=weight_threshold
                                                            )
        return mean_squared_error, poles, residues, feedthrough

    mean_squared_error, poles, residues, feedthrough = optimize_order(bias_fn, min_order, max_order)

    return poles, residues, feedthrough, mean_squared_error


def state_space_discrete(poles, residues, feedthrough):
        model_order = poles.shape[0]
        num_ports = feedthrough.shape[0]
        
        A = jnp.zeros(
            (model_order * num_ports, model_order * num_ports), dtype=complex
        )
        B = jnp.zeros((model_order * num_ports, num_ports), dtype=complex)
        C = jnp.zeros((num_ports, model_order * num_ports), dtype=complex)
        for i in range(model_order):
            A = A.at[i * num_ports : (i + 1) * num_ports, i * num_ports : (i + 1) * num_ports].set(poles[i] * jnp.eye(num_ports))
            B = B.at[i * num_ports : (i + 1) * num_ports, :].set(jnp.eye(num_ports))
            C = C.at[:, i * num_ports : (i + 1) * num_ports].set(residues[i, :, :])

        D = feedthrough
        return A, B, C, D

def state_space_step_discrete(A, B, C, D, x, u):
        x_next = A @ x + B @ u
        y = C @ x + D @ u
        return x_next, y

def state_space_response_discrete(A, B, C, D, u, x=None):
    out_samples = len(u)
    # stoptime = (out_samples) * dt

    xout = jnp.zeros((out_samples, A.shape[0]), dtype=complex)
    yout = jnp.zeros((out_samples, C.shape[0]), dtype=complex)
    # tout = jnp.linspace(0.0, stoptime, num=out_samples)

    xout = xout.at[0, :].set(jnp.zeros((A.shape[1],), dtype=complex))

    if x is not None:
        xout = xout.at[0, :].set(x)

    u_dt = u

    # Simulate the system
    for i in range(0, out_samples):
        xout = xout.at[i+1, :].set(jnp.dot(A, xout[i, :]) + jnp.dot(B, u_dt[i, :]))
        yout = yout.at[i, :].set(jnp.dot(C, xout[i, :]) + jnp.dot(D, u_dt[i, :]))

    # Last point
    yout = yout.at[out_samples - 1, :].set(jnp.dot(C, xout[out_samples - 1, :]) + jnp.dot(
        D, u_dt[out_samples - 1, :]
    ))

    return yout, xout

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
    
    poles = jnp.array([jnp.exp(1j*2*jnp.pi*center_frequency/sampling_frequency), jnp.exp(1j*3*jnp.pi/2)])
    residues = jnp.array([
        [
            [1 + 2j, 3 + 4j],
            [5 + 6j, 7 + 8j],
        ],
        [
            [1.5 + 2.5j, 3.5 + 4.5j],
            [5.5 + 6.5j, 7.5 + 8.5j]
        ],
    ])

    feedthrough = jnp.zeros((1, 1), dtype=complex)
    
    
    
    H = pole_residue_response_discrete(frequency, center_frequency, sampling_frequency, poles, residues, feedthrough)
    pass

if __name__ == "__main__":
    main()