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

xout = None
y = np.array([])
for u in sig:
    yout, xout = ring_resonator.state_space.step(u, x0=xout)
    y = np.append(y, yout[-1])

plt.plot(np.abs(y)**2)
plt.show()

ring_resonator.compute_steady_state()

# sig = np.exp(1j*2*np.pi*t*0)
# sig1 = sig.reshape(-1, 1)
# sigs = np.hstack([1*sig1, 0])

# tout, yout, xout = lsim(ring_resonator.state_space_model, sigs, t)
# plt.plot(tout, np.abs(yout[:, 0])**2)
# plt.show()
pass

