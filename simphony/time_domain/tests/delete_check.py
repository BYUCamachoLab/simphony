import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
import jax
from jax import config
config.update("jax_enable_x64", True)


from simphony.time_domain import TimeCircuit, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic,ideal

from simphony.time_domain.ideal import Modulator, MMI


import json


netlist = {
    "instances":{
        "wg": "waveguide",
        "mmi": "MultiModeInterferometer",
    },
    "connections": {
        "wg,o1": "mmi,o0",
        


    },
    "ports": {
        # "o0": "mmi,o0",
        "o0":"wg,o0",
        "o1": "mmi,o1",
        "o2": "mmi,o2",
        "o3": "mmi,o3",
        "o4":"mmi,o4",
        "o5":"mmi,o5",
        "o6": "mmi,o6",
        "o7": "mmi,o7",
        "o8": "mmi,o8",
        "o9": "mmi,o9",
        "o10": "mmi,o10",
        "o11": "mmi,o11",
        "o12": "mmi,o12",
        "o13": "mmi,o13",
        "o14": "mmi,o14",
        "o15": "mmi,o15",
        "o16": "mmi,o16",
        "o17": "mmi,o17",

        
        
        
    },
}


T = 0.5e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
MultiModeInterferometer = ideal.make_mmi_model(r=9, s=9)

models = {
    "MultiModeInterferometer": MultiModeInterferometer,
    "waveguide": ideal.waveguide,
    "coupler": ideal.coupler,
    "y_branch": siepic.y_branch,
    }
num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)

options = {
    'wl': wvl,'wg': {'length':0.0},
}


time_sim = TimeSim(netlist=netlist, models=models)
local_I = jnp.array([])
time_sim.build_model(model_parameters=options, center_wvl=1.55, dt=dt)
num_outputs = len(time_sim.netlist['ports'])
inputs = {}
for i in range(num_outputs):
    if i == 0:  
        # inputs[f'o{i}'] = gaussian_pulse(t, t0=3e-12, std=0.5e-12)
        inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
        # # inputs[f'o{i}'] = jnp.ones_like(t)
    # elif i == 1:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    #  #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 2:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 3:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 4:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 5:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 6:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 7:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    # elif i == 8:
    #     inputs[f'o{i}'] = smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10)
    else:
        inputs[f'o{i}'] = jnp.zeros_like(t)

result = time_sim.run(t, inputs)
result.plot_sim()


