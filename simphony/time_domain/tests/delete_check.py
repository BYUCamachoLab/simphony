import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
import jax
from jax import config
config.update("jax_enable_x64", True)
import os, sys
sys.path.append("/Users/mw742/simphony")
# 1. Where Python is “running” right now
print("CWD:", os.getcwd())

# 2. All the places Python will search for modules
print("sys.path:", sys.path)

from simphony.time_domain import TimeCircuit, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic,ideal
from simphony.time_domain.ideal import Modulator


import json


netlist = {
    "instances":{
        "mmi": "MultiModeInterferometer",
        # "wg": "waveguide",


    },
    "connections": {


    },
    "ports": {
        "o0": "mmi,o0",
        "o1": "mmi,o1",
        "o2": "mmi,o2",
        "o3": "mmi,o3",


        # "o0": "wg,o0",
        # "o1": "wg,o1",
    },
}


T = 1.0e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
models = {
    "MultiModeInterferometer": ideal.MMI,
    "waveguide": ideal.waveguide,
    "coupler": ideal.coupler,
    }
num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {
    'wl': wvl, 'mmi': {'wl':wvl},
}


time_sim = TimeSim(netlist=netlist, models=models)
local_I = jnp.array([])
time_sim.build_model(model_parameters=options, center_wvl=1.55, dt=dt, max_size=2, suppress_output=True)
num_outputs = len(time_sim.netlist['ports'])
inputs = {
    f'o{i}': smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10) if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}
result = time_sim.run(t, inputs)
result.plot_sim()
outputs = result.outputs
plt.plot(t, jnp.angle(outputs['o3']))
plt.show()



