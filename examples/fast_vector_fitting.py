import numpy as np
import matplotlib.pyplot as plt
import sax

from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix
from scipy.signal import  StateSpace, lsim, find_peaks, gausspulse

from simphony.fast_vector_fitting import FastVF, FVF_Options, ComputeModelResponse, real_valued_ABCD

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

order = 30
model_order = 200

options = FVF_Options()
c = 299792458
wvl_microns = np.linspace(1.5, 1.6, model_order)
omega_THz = ( c / (wvl_microns * 1e-6) )
adjustment_factor = 1
omega_THz = np.flip(omega_THz) / adjustment_factor 

s = circuit(wl=np.linspace(1.5, 1.6, model_order), wg={"length": 75.0, "loss": 100})
H = dict_to_matrix(s)
H = np.asarray(H)
model = FastVF(2*np.pi*omega_THz, H[:, :1, 1:2], order, options)

omega_model = np.linspace(min(2*np.pi*omega_THz), max(2*np.pi*omega_THz), model_order)
wvl_model = np.flip(c * 1e-6 / omega_model) * 2 * np.pi
H = dict_to_matrix(s)

Z_model = ComputeModelResponse(2*np.pi*omega_THz, model.R0, model.Rr, model.Rc, model.poles_real, model.poles_complex)

# Magnitude
plt.clf()
plt.scatter(omega_model, np.abs(H[:, 0, 1])**2)
plt.plot(omega_model, np.abs(np.squeeze(Z_model))**2, 'r-.', linewidth=1.5)
plt.legend(['Samples' ,f'Model (order={order})'])
plt.title("Magnitude")
plt.show()

# Phase
plt.clf()
plt.title("Phase")
plt.scatter(omega_model, np.angle(H[:, 0, 1]))
plt.plot(omega_model, np.angle(np.squeeze(Z_model)), 'r-.', linewidth=1.5)
plt.legend(['Samples' ,f'Model (order={order})'])
plt.show()

N = int(1e6)
T = 2e-11

A, B, C, D = real_valued_ABCD(model)
sys = StateSpace(A, B, C, np.real(D))

t = np.linspace(0, T, N)
freq_index = 120
input_frequency = omega_THz[freq_index]
u =  np.exp(1j*2*np.pi*input_frequency*t)

plt.clf()
plt.title(f"Time Response - Continuous {input_frequency}THz")
tout, yout_real, xout = lsim(sys, np.real(u), t)
tout, yout_imag, xout = lsim(sys, np.imag(u), t)
yout = yout_real + 1j*yout_imag
#plt.plot(tout, np.real(yout)**2)
#peaks, _ = find_peaks(np.real(yout)**2)
#plt.scatter(tout[peaks], (np.real(yout)**2)[peaks])
plt.plot(tout, np.abs(yout)**2)
steady_state = np.full(N, np.abs(H[freq_index, 0, 1])**2)
plt.plot(tout, steady_state, "r--", alpha=0.7)
plt.show()

# pulse = np.zeros_like(t)
# pulse[0:N//2] = 1
# pulse = pulse * u

# plt.clf()
# plt.title("Input Signal")
# plt.plot(t, pulse)
# plt.show()

# tout, yout_real, xout = lsim(sys, np.real(pulse), t)
# tout, yout_imag, xout = lsim(sys, np.imag(pulse), t)
# yout = yout_real + yout_imag*1j

# plt.plot(tout, np.abs(yout)**2)
# plt.show()