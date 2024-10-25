import numpy as np
import matplotlib.pyplot as plt
import sax
import random

from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal, siepic

from simphony.baseband_vector_fitting import Baseband_Model_SingleIO, BVF_Options, CVF_Options, CVF_Model
from scipy.signal import  StateSpace, dlsim, lsim

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

num_measurements = 100
model_order = np.arange(20, 30, 1)
# model_order = 31
wvl = np.linspace(1.5, 1.6, num_measurements)

center_wvl = 1.5493
# center_wvl = 1.58
neff = 2.34
ng = 3.4
c = 299792458
ring_length = 77.0
stepping_time = ring_length * 1e-6 / (c / ng)

S = circuit(wl=wvl, wg={"length": ring_length, "loss": 100})

options = CVF_Options(pole_spacing='log',alpha=0.01, beta=1.0, baseband=True, enforce_stability=True, poles_estimation_threshold=100, max_iterations=10, model_error_threshold = 1e-10, dt=1e-15, order=model_order, center_wvl=center_wvl)

ring_resonator = CVF_Model(wvl, S, options)
print(f"{ring_resonator.error}")
plt.scatter(ring_resonator.initial_poles().real, ring_resonator.initial_poles().imag)
plt.scatter(ring_resonator.poles.real, ring_resonator.poles.imag)
plt.show()
# ring_resonator.plot_frequency_domain_model()
ring_resonator.plot_frequency_domain_model()

N = int(15000)
T = 200.0e-12

t = np.linspace(0, T, N)
sig = np.exp(1j*2*np.pi*t*0).reshape(-1, 1)
sig = np.hstack([sig, np.zeros_like(sig)])

sig = np.exp(1j*2*np.pi*t*0)

delta = 0.01
A = 1
f = 1/(T)
width = 1.6
smoothed_square = (A/np.pi)*np.arctan(np.sin(((1 / width)*2*np.pi)*t*f - np.pi/2 + 1.2)/delta) + 0.5
sig1 = 0 * sig
sig2 = 1 * sig * smoothed_square
sig3 = 0 * sig

sig = np.hstack([sig1, sig2, sig1, sig1, sig1])

from scipy.signal import impulse, StateSpace

sys = ring_resonator.state_space.continuous
A = sys.A
B = np.array([sys.B[:, 1]])
B = B.reshape(-1, 1)
C = np.array([sys.C[1, :]])
D = np.array([sys.D[1, 1]])
sys2 = StateSpace(A, B, C, D)
t, y = impulse(sys2, N= 10000)
plt.plot(t, np.abs(y)**2)
plt.axvline(stepping_time, color="r")
plt.axvline(2 * stepping_time, color="r")
plt.axvline(3 * stepping_time, color="r")
plt.axvline(4 * stepping_time, color="r")
plt.axvline(5 * stepping_time, color="r")
plt.show()

# h = ring_resonator.state_space.discrete.to_tf()

t = np.linspace(0, T, N)
sig1 = np.exp(1j*2*np.pi*t*0).reshape(-1, 1)
sig1 = np.hstack([np.zeros_like(sig1), np.zeros_like(sig1)])

sig2 = (np.exp(1j*2*np.pi*t*0) * smoothed_square).reshape(-1, 1)
sig2 = np.hstack([sig2, np.zeros_like(sig2)])

sig = np.vstack([sig1, sig2])



xout = None
y = np.array([])
for u in sig:
    yout, xout = ring_resonator.state_space.step(u, x0=xout)
    y = np.append(y, yout[-1])

t= np.linspace()
plt.plot(t, np.abs(y)**2)
plt.plot(t, sig[0])
plt.show()

ring_resonator.compute_steady_state()

# sig = np.exp(1j*2*np.pi*t*0)
# sig1 = sig.reshape(-1, 1)
# sigs = np.hstack([1*sig1, 0])

# tout, yout, xout = lsim(ring_resonator.state_space_model, sigs, t)
# plt.plot(tout, np.abs(yout[:, 0])**2)
# plt.show()
pass

