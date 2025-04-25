import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)

from simphony.time_domain.simulation import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator

import time

T = 0.1e-13
dt = 1e-16 
dte = 5e-14     # Total time duration (40 ps)
t = jnp.arange(0, T, dte) # Time array
t0 = 0.05e-13  # Pulse start time
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

# Compute the Gaussian function
gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0*x
timePhaseInstantiated = Modulator(mod_signal=zero)

netlist={
    "instances": {
        "wg": "waveguide",
        "pm": "phase_modulator",
        "wg2": "waveguide",
    },
    "connections": {
        "wg,o1":"pm,o0",
        "pm,o1":"wg2,o0",
        # "wg,o1":"wg2,o0",
    },
    "ports": {
        "o0":"wg,o0",
        "o1":"wg2,o1",
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
    active_components=active_components,
    )

num_measurements = 200
model_order = 50
center_wvl = 1.548
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}

tic = time.time()
time_sim.build_model(model_parameters=options, dt = dt)
toc = time.time()
build_time = toc - tic

num_outputs = 2


inputs = {
    f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0   else jnp.zeros_like(t)
    for i in range(num_outputs)
}
# inputs = {
#             f'o{i}': smooth_rectangular_pulse(t, 0.0e-11,2.5e-11) if i == 0 else jnp.zeros_like(t)
#             for i in range(num_outputs)
#         }


tic = time.time()
modelResult =time_sim.run(t, inputs)
toc = time.time()
run_time = toc - tic

print(f"Build time: {build_time}")
print(f"Run time: {run_time}")
modelResult.plot_sim()



