import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from jax import config

config.update("jax_enable_x64", True)
import pickle
import time

from scipy import signal

from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
from simphony.time_domain.pole_residue_model import BVF_Options, IIRModelBaseband
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.utils import SPEED_OF_LIGHT, dict_to_matrix
# Simulation parameters
T = 100e-11
dt = 1e-14  # Time step (Total time duration is T)
t = jnp.arange(0, T, dt)  # Time array
t0 = 1.0e-11  # Pulse start time

# Modulator signals
f_mod = 0
m = f_mod * jnp.ones(len(t), dtype=complex)
f_mod2 = jnp.pi / 4
# m2 = f_mod2 * jnp.ones(len(t),dtype=complex)

x = jnp.linspace(0, 3.14, len(t))
mu = 1.30  # center the Gaussian in the middle of the interval
sigma = 0.15  # adjust sigma for desired width

# Compute the Gaussian function
gaussian = np.pi * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
# gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0 * x
timePhaseInstantiated = Modulator(mod_signal=gaussian)

# Define netlist and models
netlist = {
    "instances": {
        "wg1": "waveguide",
        "wg2": "waveguide",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "wg5": "waveguide",
        "wg6": "waveguide",
       
    },
    "connections": {
    #    "wg1,o1": " wg2,o0",
        # "wg2,o1": "wg3,o0",
        # "wg3,o1": "wg4,o0",
        # "wg4,o1": "wg5,o0",
        # "wg5,o1": "wg6,o0",
        # "wg6,o1": "wg7,o0",
        # "wg7,o1": "wg8,o0",
    },
    "ports": {
        "o0": "wg1,o0",
        "o1": "wg1,o1",
    },
}

models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
}

active_components = {"pm", "pm2"}
num_measurements = 200

center_wvl = 1.55
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {
    "wl": wvl,
    "wg1": {"length": 290.0, "loss": 0},
}

# Create and build simulation
time_sim = TimeSim(
    netlist=netlist,
    models=models,
    settings=options,
)
result = time_sim.run(t, {
    "o0": smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11),
    "o1" : jnp.zeros_like(t),
}, carrier_freq=SPEED_OF_LIGHT/(center_wvl*1e-6), dt=dt)

result.plot_sim()   

# circuit, _ = sax.circuit(
#                             netlist=netlist,
#                             models=models,
#                         )

# s_params_dict = circuit(**options)
# s_matrix = np.asarray(dict_to_matrix(s_params_dict))
# c_light = 299792458
# center_freq = c_light / (center_wvl * 1e-6)

# freqs = c_light / (wvl * 1e-6) - center_freq
# sampling_freq = -1 / dt
# beta = sampling_freq / (freqs[-1] - freqs[0])
# bvf_options = BVF_Options(beta=beta,max_iterations = 30)
# sorted_ports = sorted(netlist["ports"].keys(), key=lambda p: int(p.lstrip('o')))
# freqs_hz = SPEED_OF_LIGHT / (wvl * 1e-6)    # wvl was in μm → convert to m

# omega = 2 * np.pi * freqs_hz   
# idx         = np.argsort(omega)
# omega_s     = omega[idx]
# group_delay = -np.gradient(np.unwrap(np.angle(s_matrix[:, 0, 1])),omega_s)
# plt.plot(wvl, group_delay*1e12)              # convert s → ps
# plt.xlabel("Angular frequency ω (rad/s)")
# plt.ylabel("Group delay")
# plt.show() 
# iir_model = IIRModelBaseband(
#     wvl, center_wvl, s_matrix,order = 80, options=bvf_options
# )

# poles = iir_model.poles
# residues = iir_model.residues
# D = iir_model.D
# Ω = 2*np.pi * freqs / sampling_freq   

# z = np.exp(1j * Ω)

# S_fit = np.zeros_like(z, dtype=complex)
# for i, p in enumerate(poles):
#     r01    = residues[i,0,1]
#     S_fit += r01 / (z - p)
# S_fit += D[0,1]
# print(residues[:,0,1])
# print(poles)
# plt.plot(wvl, np.angle(s_matrix[:, 0, 1]), label='IIR Model')
# plt.plot(wvl, np.angle(S_fit), label='IIR Model')
# plt.show()

# plt.plot(wvl, np.abs(s_matrix[:, 0, 1]), label='IIR Model')
# plt.plot(wvl, np.abs(S_fit), label='IIR Model Fit')
# plt.show()

# plt.figure(figsize=(5,5))
# # unit circle
# angle = np.linspace(0, 2*np.pi, 400)
# plt.plot(np.cos(angle), np.sin(angle), 'gray', lw=1)

# plt.scatter(poles.real, poles.imag, marker='x', s=80, label='Poles')

# plt.axhline(0, color='black', lw=1)
# plt.axvline(0, color='black', lw=1)
# plt.xlabel('Re\{z\}')
# plt.ylabel('Im\{z\}')
# plt.title('Pole–Zero Map (z-plane)')
# plt.legend()
# plt.axis('equal')
# plt.grid(True, ls='--', alpha=0.5)
# plt.show()


