import os
import pickle
import pytest
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
from simphony.time_domain.simulation import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
T = 2.0e-11
dt = 0.5e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt)

netlist={
    "instances": {
        "wg": "waveguide",
        "wg2": "waveguide",
        "y": "y_branch",
        "hr":"half_ring",
        "hr2":"half_ring",
        "y2": "y_branch",
    },
    "connections": {
        "hr,port_1":"hr2,port_1",
        "hr,port_3":"hr2,port_3",

    },
    "ports": {
        "o0":"hr,port_2",
        "o2":"hr2,port_4",
        "o1":"hr,port_4",
        "o3":"hr2,port_2",
    },
}
models={
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "half_ring": siepic.half_ring,
    "bidirectional": siepic.bidirectional_coupler,
}



time_sim = TimeSim(
    netlist=netlist,
    models=models,
    
    )

num_measurements = 200
model_order = 50
center_wvl = 1.548
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100}}


time_sim.build_model(model_parameters=options, dt = dt,max_size = 3)



num_outputs = 4

inputs = {
            f'o{i}': smooth_rectangular_pulse(t,0.0e-11,4.0e-11) if i == 0 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }


modelResult =time_sim.run(t, inputs)
modelResult.plot_sim()