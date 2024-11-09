import numpy as np
import matplotlib.pyplot as plt
import sax

from jax import config
config.update("jax_enable_x64", True)


from simphony.libraries import ideal
from simphony.utils import dict_to_matrix

from simphony.baseband_vector_fitting import Baseband_Model, BVF_Options
from scipy.signal import  StateSpace, dlsim, ss2tf, find_peaks

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
)x  

num_measurements = 100
wvl = np.linspace(1.5, 1.6, num_measurements)

s = circuit(wl=wvl, wg={"length": 75.0, "loss": 100})
S = dict_to_matrix(s)
S = np.asarray(S)
model_order = 35
center_wvl = 1.548
options = BVF_Options(max_iterations=5, beta=7.0, gamma=0.95) # 3 iterations
model = Baseband_Model(wvl, center_wvl, S[:, 0, 1], model_order, options)

model.fit_model()
# model.plot_poles()
# plt.show()
response = model.compute_response()
peaks, _ = find_peaks(-np.abs(response)**2)
plt.scatter(model.freqs[peaks], np.abs(response[peaks])**2)
plt.plot(model.freqs, np.abs(response**2))
plt.show()

c = 299792458
print(np.diff(model.freqs[peaks]))
fsr = np.diff(model.freqs[peaks])
ng = c / (fsr * 75e-6)

print(ng)

dL_microns = 75.0
roundtrip_time = dL_microns * 1e-6 / (c / 3.4)

A, B, C, D = model.real_ABCD_matrices()
sys = StateSpace(A, B, C, D, dt=1/model.sampling_freq)

N = int(10000)
T = 5e-11
t = np.linspace(0, T, N)

sig = np.exp(1j*2*np.pi*t*0)
sig[:N//2] *= 1
sig[N//2:] *= 0
#sig = np.full(t.shape, 1.0)

tout, yout = model.compute_time_response(sig=sig, t=t)
model.plot_time_response()
plt.show()