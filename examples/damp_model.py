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
from simphony.baseband_vector_fitting import BasebandModel, DampModel

from scipy.signal import lsim

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

wvl_microns = np.linspace(1.51, 1.59, 200)
center_wvl = 1.55

ckt = circuit(wl=wvl_microns, wg={"length": 100, "loss": 50})
s_params = np.copy(np.asarray(dict_to_matrix(ckt)))


T = 30e-12
# t = np.linspace(0.0, T, 1000)
baseband_model  = BasebandModel(wvl_microns, center_wvl, s_params, 50)
damp_model = DampModel(baseband_model, T)
K=damp_model.K

n = damp_model.num_modes
input_signal = np.zeros((K, n), dtype=complex)
# input_signal2plt = np.zeros((K, n), dtype=complex)
input_signal[:, 0] = smooth_rectangular_pulse(damp_model.t, 10e-13, 200e-13)
# input_signal2plt[:, 0] = smooth_rectangular_pulse(damp_model.t, 10e-13, 20e-13)
# input_signal[:, 0] = smooth_rectangular_pulse(damp_model.t, 21e-13, 31e-13)
# input_signal[:, 0] = 1.0 + 0.0*1j
# input_signal2plt[:, 1] = 0.0 + 0.0*1j
input_signal[:, 1] = 0.0 + 0.0*1j

damp_model.calculate_damps()
t, responses = damp_model.compute_impulse_responses(input_signal, 1, 0)
cummulative_response = np.sum(responses, axis=0)
plt.plot(t, np.abs(cummulative_response)**2, label="Impulse response")
plt.plot(t, np.abs(input_signal[:, 0])**2, label="Input Signal")
plt.xlabel("Time")
plt.ylabel("E-Field Amp")
plt.legend()
plt.show()

from scipy.signal import find_peaks

width = 225
for i in np.arange(1, 30, 1):
    # plt.plot(t, np.abs(responses[400 + i*width, :])**2, linewidth=5.4, alpha = 0.4)
    peaks, _ = find_peaks(np.abs(responses[400 + i*width, :])**2)
    plt.scatter(t[peaks], np.abs(responses[400 + i*width, peaks])**2, marker='o', s=10)


plt.plot(t[0:7000], np.abs(cummulative_response[0:7000])**2, label="Impulse response")

plt.show()



plt.plot(np.abs(responses[2500, :])**2)
plt.plot(np.abs(responses[2510, :])**2)
plt.plot(np.abs(responses[2520, :])**2)
plt.plot(np.abs(responses[2530, :])**2)
plt.plot(np.abs(responses[2540, :])**2)
plt.plot(np.abs(responses[2550, :])**2)
plt.plot(np.abs(responses[2560, :])**2)
plt.plot(np.abs(responses[2570, :])**2)
plt.plot(np.abs(responses[2580, :])**2)
plt.plot(np.abs(responses[2590, :])**2)
plt.plot(np.abs(responses[2600, :])**2)
plt.plot(np.abs(responses[2610, :])**2)
plt.show()

# t, r = damp_model.compute_response(input_signal)
# plt.plot(t, np.abs(r[:, 1, 0])**2)
# plt.plot(t, np.abs(input_signal)**2)
# plt.show()

test_wls = np.linspace(1.52, 1.57, 100)
steady_state = np.array([])
for wl0 in test_wls:
    print(wl0)
    # plt.scatter(np.linspace(test_wls[0], wl0, len(steady_state)), np.abs(steady_state)**2)
    # plt.plot(wvl_microns, np.abs(s_params[:, 0, 1])**2)
    baseband_model  = BasebandModel(wvl_microns, wl0, s_params, 30)
    damp_model = DampModel(baseband_model, T)
    damp_model.calculate_damps()
    t, response = damp_model.compute_response(input_signal)
    # plt.plot(t, np.abs(response[:, 0, 1])**2)
    # plt.show()
    steady_state = np.append(steady_state, response[-1, 0, 1])


plt.scatter(test_wls, np.abs(steady_state)**2)
plt.plot(wvl_microns, np.abs(s_params[:, 0, 1])**2)
plt.show()

pass
