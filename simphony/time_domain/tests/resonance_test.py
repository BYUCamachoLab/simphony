import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)
from scipy import signal
from simphony.utils import dict_to_matrix
from simphony.time_domain.simulation import TimeSim, TimeResult
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
from simphony.libraries import ideal

import time

# time‐array setup
T = 0.25e-10
dt = 1e-14
t = jnp.arange(0, T, dt)
num_outputs = 3

# static parts of the netlist & models
netlist = {
    "instances": {
        "dc":"directional_coupler",
        "pm":"phase_modulator",
        "wg":"waveguide",
        "wg2":"waveguide",
        "dc2":"directional_coupler",
        "t":"terminator",
    },
    "connections": {
        "dc,port_3":"wg,o0",
        "pm,o0":"wg,o1",
        "pm,o1":"dc2,port_1",
        "wg2,o0":"dc2,port_3",
        "wg2,o1":"dc,port_1",
        "dc2,port_2":"t,port_1",
    },
    "ports": {
        "o0":"dc,port_2",
        "o1":"dc,port_4",
        "o2":"dc2,port_4",
    },
}
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    # placeholder for Modulator, we’ll reassign below inside the loop
    "phase_modulator": None,
    "half_ring": siepic.half_ring,
    "terminator": siepic.terminator,
    "directional_coupler": siepic.directional_coupler,
    "coupler": ideal.coupler,
}
active_components = {"pm"}

# build-time options
center_wvl = 1.55
wvl = np.linspace(1.50, 1.60, 200)
options = {
    'wl': wvl,
    'wg': {"length": 50.0, "loss": 100},
    'wg2':{"length": 10.0, "loss": 100},
    'dr':{"coupling_length": 12.5},
    'dr2':{"coupling_length": 12.5},
}

# Prepare your input once
inputs = {
    f'o{i}': smooth_rectangular_pulse(t, 0.01e-10, 0.25e-10) if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}

# The loop over phases
phases = np.linspace(-2*np.pi, 2*np.pi, 50)
results = []

for phi in phases:
    # instantiate the modulator with this phase
    m2 = phi * jnp.ones(len(t), dtype=complex)
    models["phase_modulator"] = Modulator(mod_signal=m2)

    # build & run the sim
    time_sim = TimeSim(
        netlist=netlist,
        models=models,
        active_components=active_components,
    )
    time_sim.build_model(
        model_parameters=options,
        center_wvl=center_wvl,
        dt=dt,
        max_size=3
    )

    # one warm-up run (optional)
    _ = time_sim.run(t, inputs)

    # actual run
    modelResult: TimeResult = time_sim.run(t, inputs)

    # grab the final value of output port o0 (for instance)
    last_val = np.array(modelResult.outputs['o0'])[-1]
    results.append(last_val)

# now `results[i]` is the final o0 output for phase=phases[i]
# you can plot it:
plt.plot(phases, np.real(results), label='Re(o0 last)')
#plt.plot(phases, np.imag(results), label='Im(o0 last)')
plt.xlabel('Applied phase (rad)')
plt.ylabel('Final output signal')
plt.legend()
plt.show()
