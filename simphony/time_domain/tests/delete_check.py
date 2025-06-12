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
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse

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
        "wg": "waveguide",
        "wg2": "waveguide",
        "pm": "phase_modulator",
        # "pm2": "phase_modulator",
        "y": "y_branch",
        "y2": "y_branch",
    },
    "connections": {
        "wg,o0": "y,port_2",
        "wg,o1": "pm,o0",
        # "pm,o1":"pm2,o0",
        # "y2,port_2": "pm2,o1",
        "y2,port_2": "pm,o1",
        "wg2,o0": "y,port_3",
        "y2,port_3": "wg2,o1",
    },
    "ports": {
        "o0": "y,port_1",
        "o1": "y2,port_1",
    },
}
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
}
active_components = {"pm", "pm2"}
num_measurements = 200
model_order = 50
center_wvl = 1.548
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {
    "wl": wvl,
    "wg": {"length": 10.0, "loss": 100},
    "wg2": {"length": 10.0, "loss": 100},
}

# Create and build simulation
time_sim = TimeSim(
    netlist=netlist,
    models=models,
    model_settings=options,
)
# modelResult = time_sim.run(t, {"o0": jnp.ones_like(t), "o1": jnp.zeros_like(t)})

# modelResult.plot_sim()
new_netlist = {
    "instances": {
        "wg": "waveguide",
        "wg2": "waveguide",
        "pm": "phase_modulator_time",
        "y": "y_branch",
        "y2": "y_branch",
        "time": "time_system",
    },
    "connections": {
        "wg,o0": "y,port_2",
        "wg,o1": "pm,o0",
        "y2,port_2": "pm,o1",
        "wg2,o0": "y,port_3",
        "y2,port_3": "wg2,o1",
        "y2,port_1": "time,o0",
    },
    "ports": {
        "o0": "y,port_1",
        "o1": "time,o1",
    },
}


f_mod = 0
m = f_mod * jnp.ones(len(t), dtype=complex)
f_mod2 = jnp.pi / 4
# m2 = f_mod2 * jnp.ones(len(t),dtype=complex)

x = jnp.linspace(0, 3.14, len(t))
mu = 0.5  # center the Gaussian in the middle of the interval
sigma = 0.15  # adjust sigma for desired width

# Compute the Gaussian function
gaussian = np.pi * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
# gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0 * x
timePhaseInstantiated1 = Modulator(mod_signal=gaussian)


models["time_system"] = time_sim
models["phase_modulator_time"] = timePhaseInstantiated1
time_simmer2 = TimeSim(
    netlist=new_netlist,
    models=models,
    model_settings=options,
)

new_netlist_2 = {
    "instances": {
        "wg": "waveguide",
        "wg2": "waveguide",
        "pm": "phase_modulator_time2",
        "y": "y_branch",
        "y2": "y_branch",
        "time2": "time_system2",
        "time": "time_system",
    },
    "connections": {
        "wg,o0": "y,port_2",
        "wg,o1": "pm,o0",
        "y2,port_2": "pm,o1",
        "wg2,o0": "y,port_3",
        "y2,port_3": "wg2,o1",
        "y2,port_1": "time2,o0",
        "time,o0": "time2,o1",
    },
    "ports": {
        "o0": "y,port_1",
        "o1": "time,o1",
    },
}
f_mod = 0
m = f_mod * jnp.ones(len(t), dtype=complex)
f_mod2 = jnp.pi / 4
# m2 = f_mod2 * jnp.ones(len(t),dtype=complex)

x = jnp.linspace(0, 3.14, len(t))
mu = 2.0  # center the Gaussian in the middle of the interval
sigma = 0.15  # adjust sigma for desired width

# Compute the Gaussian function
gaussian = np.pi * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
# gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0 * x
timePhaseInstantiated2 = Modulator(mod_signal=gaussian)


models["time_system2"] = time_simmer2
models["phase_modulator_time2"] = timePhaseInstantiated2
time_simmer3 = TimeSim(
    netlist=new_netlist_2,
    models=models,
    model_settings=options,
)
num_outputs = 2
inputs = {
    f"o{i}": jnp.ones_like(t) if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}

modelResult = time_simmer3.run(t, inputs, carrier_freq=193e12, dt=dt)

modelResult.plot_sim()
