from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sax
import pandas as pd

from simphony.quantum import QuantumTimeElement
from simphony.libraries import siepic, ideal
from simphony.utils import smooth_rectangular_pulse, dict_to_matrix, gaussian_pulse
from simphony.baseband_vector_fitting import BasebandModel
from scipy.signal import find_peaks

from scipy.signal import lsim

# circuit, info = sax.circuit(
#     netlist= {
#         "instances": {
#             "cp1": "coupler",
#         },
#         "connections": {
#         },
#         "ports": {
#             "o0": "cp1,port 1",
#             "o1": "cp1,port 2",
#             "o2": "cp1,port 3",
#             "o3": "cp1,port 4",
#         }
#     },
#     models={
#         "coupler": siepic.directional_coupler,
#     }
# )

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

# netlist = {
#     "instances": {
#         "wg": "waveguide",
#         "hr": "half_ring",
#     },
#     "connections": {
#         "hr,port 3": "wg,o0",
#         "hr,port 2": "wg,o1",
#     },
#     "ports": {
#         "o0": "hr,port 1",
#         "o1": "hr,port 4",
#     }
# }

# circuit, info = sax.circuit(
#     netlist=netlist,
#     models={
#         "waveguide": siepic.waveguide,
#         "half_ring": siepic.bidirectional_coupler,
#     }
# )

wvl_microns = np.linspace(1.50, 1.60, 200)
center_wvl = 1.55

ckt = circuit(wl=wvl_microns, wg={"length": 50, "loss": 50})
s_params = np.copy(np.asarray(dict_to_matrix(ckt)))

def mydlsim(system, u, t=None, x0=None):
    out_samples = len(u)
    stoptime = (out_samples - 1) * system.dt

    xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
    yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
    tout = np.linspace(0.0, stoptime, num=out_samples)

    xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)

    u_dt = u

    # Simulate the system
    for i in range(0, out_samples - 1):
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]))
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))

    # Last point
    yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                              np.dot(system.D, u_dt[out_samples-1, :]))

    return tout, yout, xout



model = BasebandModel(wvl_microns, center_wvl, s_params, 30)
sys = model.generate_sys_discrete()

plt.plot(wvl_microns, np.abs(s_params[:, 0, 1]))
plt.plot(wvl_microns, np.abs(model.compute_response(0, 1)))
plt.show()

steady_state = np.array([])
wls = np.array([])

test_wls = np.linspace(1.52, 1.57, 80)
for wl0 in test_wls:
    print(wl0)
    model = BasebandModel(wvl_microns, wl0, s_params, 32)
    sys = model.generate_sys_discrete()

    # plt.plot(wvl_microns, np.abs(s_params[:, 0, 1]))
    # plt.plot(wvl_microns, np.abs(model.compute_response(0, 1)))
    # plt.show()

    dt = sys.dt
    n = sys.B.shape[1]
    K = 15000 # number of time steps
    # K = 500 # number of time steps
    t = np.linspace(0, K*dt, K)

    input_signal = np.zeros((K, n), dtype=complex)
    input_signal[:, 0] = 1.0 + 0.0*1j
    input_signal[:, 1] = 0.0 + 0.0*1j
    t, orig_yout, x = mydlsim(sys, input_signal)
    # plt.plot(t, np.abs(orig_yout)**2)
    # plt.show()
    steady_state = np.append(steady_state, orig_yout[-1, 1])
    wls = np.append(wls, wl0)

plt.plot(wvl_microns, np.abs(model.compute_response(0, 1))**2)
plt.plot(test_wls, np.abs(steady_state)**2)
# plt.plot(wvl_microns, np.abs(s_params[:, 0, 1])**2)
# plt.plot(wvl_microns, np.abs(s_params[:, 0, 1])**2)
# plt.plot(wls, np.abs(steady_state)**2)
plt.show()
    


dt = sys.dt
n = sys.B.shape[1]
K = 7500 # number of time steps
# K = 500 # number of time steps
t = np.linspace(0, K*dt, K)

input_signal = np.zeros((K, n), dtype=complex)
input_signal[:, 0] = 1.0 + 0.0*1j
input_signal[:, 1] = 0.0 + 0.0*1j
t, orig_yout, x = mydlsim(sys, input_signal)
plt.plot(t, np.abs(orig_yout)**2)
plt.show()

_y1 = []
for k in range((3*K)//4):
    pulse = smooth_rectangular_pulse(t, t_start=(k+1)*dt*100, t_end=(k+2)*dt*100)

    input_signal = np.zeros((K, n), dtype=complex)
    input_signal[:, 0] = pulse
    input_signal[:, 1] = 0.0 + 0.0*1j

    t, y, x = mydlsim(sys, input_signal)
    _y1.append(y)
    plt.plot(t, np.abs(input_signal)**2, alpha=0.75)

_y2 = []
for k in range((3*K)//4):
    pulse = gaussian_pulse(t, ((k+1)*dt*100 + (k+2)*dt*100)/2, 4e-13)

    input_signal = np.zeros((K, n), dtype=complex)
    input_signal[:, 0] = pulse
    input_signal[:, 1] = 0.0 + 0.0*1j

    t, y, x = mydlsim(sys, input_signal)
    _y2.append(y)
    plt.plot(t, np.abs(input_signal)**2)
plt.show()

sig = np.zeros((K, n))

for y in _y1[0:10]:
    plt.plot(t, np.abs(y)**2, alpha=0.4)
    plt.plot(t, np.angle(y))

for y in _y2[0:10]:
    plt.plot(t, np.abs(y)**2)
plt.show()

for y in _y1: 
    sig = sig + y

plt.show()



plt.plot(t, np.abs(sig)**2)
plt.show()

plt.plot(t, np.abs(input_signal)**2)
plt.plot(t, np.abs(y)**2)
plt.show()
pass

# def compute_covariance(K, n, input_signal):
#     mean_real = 0
#     std_real = np.sqrt(0.25)
#     mean_imag = 0
#     std_imag = np.sqrt(0.25)

#     # Generate Gaussian noise for real and imaginary parts
#     noise_real = np.random.normal(mean_real, std_real, size=(K, n))
#     noise_imag = np.random.normal(mean_imag, std_imag, size=(K, n))
#     # Add noise to the input signal
#     noisy_input_signal = np.real(input_signal) + noise_real + 1j * (np.imag(input_signal) + noise_imag)

#     # Process the noisy signal through the black-box system
#     t, y, x = mydlsim(sys, noisy_input_signal)
#     # y = noisy_input_signal

#     # mean_real_out = 0
#     # std_real_out = np.sqrt(0.25)
#     # mean_imag_out = 0
#     # std_imag_out = np.sqrt(0.25)

#     # noise_real_out = np.random.normal(mean_real_out, std_real_out, size=(K, n))
#     # noise_imag_out = np.random.normal(mean_imag_out, std_imag_out, size=(K, n))

#     # y = np.real(y) + noise_real_out + 1j*(np.imag(y) + noise_imag_out)

#     output_signal = y[:, 1]

#     return output_signal
#     # # Extract real and imaginary parts
#     # real_quadratures = np.real(output_signal)  # X quadratures
#     # imag_quadratures = np.imag(output_signal)  # P quadratures

#     # combined_quadratures = np.vstack((real_quadratures, imag_quadratures))

#     # # Estimate the full covariance matrix
#     # full_covariance = np.cov(combined_quadratures)


# y = []
# for _ in range(500):
#     print(_)
#     y.append(compute_covariance(K, n, input_signal))

# y = np.stack(y)
# t = np.linspace(0, 1, y.shape[1])

# for yout in y:
#     plt.scatter(t, np.abs(yout)**2)
# plt.show()

# y_last = y[:, -1]
# X = np.real(y_last)
# P = np.imag(y_last)



# r = np.stack((X, P), axis=0)
# cov = np.cov(r)
# print(cov)
# print(f"var of X: {np.var(X)}")
# print(f"var of P: {np.var(P)}")





# pass
# # y_last = y[:, 0]
# # X = np.real(y_last)
# # P = np.imag(y_last)

# # r = np.stack((X, P), axis=0)
# # cov = np.cov(r)
# # print(cov)

# []

# pass




# # T = K
# # c = 299792458
# # pulse = gaussian_pulse(t, 2400*dt, std=100*dt, a=5.0)
# # input_signal = np.zeros((T, n), dtype=complex)
# # input_signal[:, 0] = pulse
# # input_signal[:, 1] = 0.0 + 0.0*1j

# # t, y, x = mydlsim(sys, input_signal)
# # # plt.plot(t, np.abs(input_signal[:, 0])**2)
# # plt.plot(t, np.abs(pulse)**2)
# # plt.plot(t, np.abs(y[:, 1])**2)
# # plt.show()