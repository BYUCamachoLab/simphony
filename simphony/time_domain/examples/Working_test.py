import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)

from simphony.time_domain.TimeSim import TimeSim, plot_time_result
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator

import time

T = 10e-11
dt = 1e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt)  # Time array
t0 = 1e-11  # Pulse start time
std = 1e-12

f_mod = 3*jnp.pi/2
m = f_mod * jnp.ones(len(t),dtype = complex)
x = jnp.linspace(0, 3.14, len(t))

# Define Gaussian parameters
mu = 1.14  # center the Gaussian in the middle of the interval
sigma = 0.3     # adjust sigma for desired width

# Compute the Gaussian function
gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0*x
timePhaseInstantiated = Modulator(mod_signal=gaussian)

netlist={
    "instances": {
        "wg": "waveguide",
        "y": "y_branch",
        "pm": "phase_modulator",
        "pm2": "phase_modulator",
        "y2": "y_branch",
        "wg2": "waveguide",
        "y3": "y_branch",
        "y4": "y_branch",
        "y5": "y_branch",
        "y6": "y_branch",
        "wg3": "waveguide",
        "wg4": "waveguide",
        "wg5": "waveguide",
        "wg6": "waveguide",
        "bdc": "bidirectional",
        "bdc2": "bidirectional",
        "bdc3": "bidirectional",
    },
    "connections": {
        "y,port_3":"wg,o0",
        "y,port_2":"wg2,o0",
        "wg2,o1":"pm,o0",
        # # "y2,port_3":"wg,o1",
        "wg,o1":"bdc,port_2",
        "bdc,port_3":"wg3,o0",
        "wg3,o1":"bdc,port_1",
        # "bdc,port_4":"y2,port_3",
        "bdc2,port_1":"bdc,port_4",
        "bdc2,port_2": "pm,o1",
        # "wg2,o1":"pm,o0",



        # "pm,o1":"y2,port_2"
        # "pm,o1": "bdc2,port_2"

    },
    "ports": {
        "o0": "y,port_1",
        "o1": "y2,port_1",

        # "o1": "bdc,port_4",
        # "o2": "pm,o1",
        # "o2":"bdc2,port_4"
        

        "o0": "y,port_1",
        "o1": "bdc2,port_3",
        "o2": "bdc2,port_4",
    },
}
models={
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
}
active_components = {
    "pm","pm2"
}


time_sim = TimeSim(
    netlist=netlist,
    models=models,
    active_components=active_components,
    )

num_measurements = 200
model_order = 100
center_wvl = 1.548
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 60.0, "loss": 100},'wg3':{"length":50.0, "loss":100}}

tic = time.time()
time_sim.build_model(model_parameters=options)
toc = time.time()
build_time = toc - tic
from math import log2, sqrt

num_outputs = 3


inputs = {
    f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}
# inputs = {
#             f'o{i}': smooth_rectangular_pulse(t,0.5e-11,3.5e-11) if i == 0 else jnp.zeros_like(t)
#             for i in range(num_outputs)
#         }


tic = time.time()
modelResult =time_sim.run(t, inputs)
toc = time.time()
run_time = toc - tic

print(f"Build time: {build_time}")
print(f"Run time: {run_time}")

plot_time_result(modelResult)

plt.plot(t, m)
plt.show()


