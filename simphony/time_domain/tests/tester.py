import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)
from scipy import signal

from simphony.time_domain.simulation import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator

import time

T = 1.0e-10
dt = 1e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt) # Time array
t0 = 1.0e-11  # Pulse start time
std = 1e-12
inter = 250

f_mod =0
m = f_mod * jnp.ones(len(t),dtype = complex)
f_mod2 =jnp.pi/4 
# m2 = f_mod2 * jnp.ones(len(t),dtype = complex)

x = jnp.linspace(0, 3.14, len(t))

# Define Gaussian parameters
mu = 1.14  # center the Gaussian in the middle of the interval
sigma = 0.3     # adjust sigma for desired width
num_blocks = 6           # number of transitions
hold_time  = 1000          # hold 30 samples at each level
amp_lo     = 0.0
amp_hi     = 0.5



# Compute the Gaussian function
gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
#gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0*x
timePhaseInstantiated = Modulator(mod_signal=zero)

netlist={
    "instances": {
        "wg": "waveguide",
        "pm": "phase_modulator",
        "wg2": "waveguide",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "wg5": "waveguide",
    },
    "connections": {

        # "wg,o0":"pm,o1",
        "wg,o1":"wg2,o0",

        "wg2,o1":"wg3,o0",
        "wg3,o1":"wg4,o0",
        "wg4,o1":"wg5,o0",
    },
    "ports": {
        "o0":"wg,o0",
        # "o0":"pm,o0",
        "o1":"wg5,o1",
    },
}
models={
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
}
active_components = {
    "pm", "pm2"
}


time_sim = TimeSim(
    netlist=netlist,
    models=models,
    
    )

num_measurements = 200
model_order = 50
center_wvl = 1.548
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {'wl':wvl,'wg':{"length": 50.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}

tic = time.time()
time_sim.build_model(model_parameters=options, dt = dt, max_size= 2)
toc = time.time()
build_time = toc - tic

num_outputs = 2


# inputs = {
#     f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0   else jnp.zeros_like(t)
#     for i in range(num_outputs)
# }

inputs = {
            f'o{i}': smooth_rectangular_pulse(t, 0.3e-10, 0.6e-10) if i == 0 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }


tic = time.time()
modelResult =time_sim.run(t, inputs)
toc = time.time()
run_time = toc - tic

print(f"Build time: {build_time}")
print(f"Run time: {run_time}")

modelResult.plot_sim()


