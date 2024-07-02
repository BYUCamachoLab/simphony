import numpy as np
import matplotlib.pyplot as plt
import sax

from jax import config
config.update("jax_enable_x64", True)


from simphony.libraries import ideal
from simphony.utils import dict_to_matrix

from simphony.baseband_vector_fitting import Baseband_Model, BVF_Options
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

num_measurements = 1000
wvl = np.linspace(1.5, 1.6, num_measurements)

s = circuit(wl=wvl, wg={"length": 75.0, "loss": 100})
S = dict_to_matrix(s)
S = np.asarray(S)
model_order = 35
center_wvl = 1.51
options = BVF_Options(max_iterations=5, beta=2.0, gamma=0.95) # 3 iterations
model = Baseband_Model(wvl, center_wvl, S[:, 0, 1], model_order, options)

model.fit_model()
model.plot_poles()
plt.show()

A, B, C, D = model.real_ABCD_matrices()
sys = StateSpace(A, B, C, D, dt = 1/model.sampling_freq)


input_frequency = 0
N = int(2)
T = 2e-11
t = np.linspace(0, T, N)


u = np.full_like(t, 1)
u = u.reshape(-1, 1)
impulse = np.hstack([np.real(u), np.imag(u)])
plt.plot(t, u)
plt.show()
t_out, yout, _ = dlsim(sys, impulse, t)
plt.plot(t_out, np.abs(yout[:, 0] + 1j*yout[:, 1])**2)
plt.show()
pass
