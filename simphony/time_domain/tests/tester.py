import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sax
from jax import config

config.update("jax_enable_x64", True)
import time

from scipy import signal

from simphony.libraries import ideal, siepic
from simphony.time_domain.ideal import Modulator
from simphony.time_domain.simulation import TimeResult, TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse
from simphony.utils import dict_to_matrix

T = 1.5e-11
dt = 1e-14  # Total time duration (40 ps)
t = jnp.arange(0, T, dt)  # Time array
t0 = 1.0e-11  # Pulse start time
std = 1e-12
inter = 250
c = 299792458
f_mod = 0
m = f_mod * jnp.ones(len(t), dtype=complex)
f_mod2 = jnp.pi / 2
m2 = f_mod2 * jnp.ones(len(t), dtype=complex)

x = jnp.linspace(0, 3.14, len(t))

# Define Gaussian parameters
mu = 1.14  # center the Gaussian in the middle of the interval
sigma = 0.3  # adjust sigma for desired width
num_blocks = 6  # number of transitions
hold_time = 1000  # hold 30 samples at each level
amp_lo = 0.0
amp_hi = 0.5


# Compute the Gaussian function
gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Optionally, normalize so the area under the curve is 1
# gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0 * x
timePhaseInstantiated = Modulator(mod_signal=m2)

netlist = {
    "instances": {
        "dc": "directional_coupler",
        # "bi":"bidirectional",
        "pm": "phase_modulator",
        "wg": "waveguide",
        # "wg2":"waveguide",
        # "dc2":"directional_coupler",
        # "t":"terminator",
    },
    "connections": {
        "dc,port_3": "wg,o0",
        # "wg,o1":"dc2,port_1",
        "pm,o0": "wg,o1",
        # "pm,o1":"dc2,port_1",
        # "wg2,o0":"dc2,port_3",
        # "wg2,o1":"dc,port_1",
        # "dc2,port_2":"t,port_1",
        "pm,o1": "dc,port_1",
        # "bi, port_1":"wg,o0",
        # "wg,o1":"pm,o1",
        # "bi, port_3":"pm,o0",
        # "bi,port_3":"wg,o1",
    },
    "ports": {
        # "o0":"dc, port_1",
        # "o1":"dc, port_2",
        # "o2":"dc, port_3",
        # "o3":"dc, port_4",
        "o0": "dc, port_2",
        "o1": "dc, port_4",
        # "o2":"dc2, port_4",
    },
}
models = {
    "waveguide": siepic.waveguide,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
    "directional_coupler": siepic.directional_coupler,
}
active_components = {"pm", "pm2"}
# circuit, _ = sax.circuit(
#                         netlist=netlist,
#                         models=models,
#                     )


# time_sim = TimeSim(
#     netlist=netlist,
#     models=models,
#     active_components=active_components,
# )

num_measurements = 200
model_order = 50
center_wvl = 1.55
wvl = np.linspace(1.50, 1.60, num_measurements)
options = {
    "wl": wvl,
    "wg": {"length": 100.0, "loss": 100},
    "wg2": {"length": 10.0, "loss": 100},
    "dr": {"coupling_length": 10},
    "dr2": {"coupling_length": 12.5},
}

# s_params_dict = circuit(**options)
# s_matrix = np.asarray(dict_to_matrix(s_params_dict))


# plt.figure()
# plt.plot(wvl, jnp.abs(s_matrix[:, 0, 2])**2, label='S11')
# plt.xlabel('Wavelength (µm)')
# plt.ylabel('S11 Parameter')
# plt.title('S11 vs Wavelength')
# plt.legend()
# plt.grid()
# plt.show()
range_of_phases = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 50)
phases_at_o1 = []
phases_at_o2 = []
time_sim = TimeSim(
    netlist=netlist,
    models=models,
    active_components=active_components,
)
num_outputs = 2
inputs = {
    f"o{i}": (
        smooth_rectangular_pulse(t, 0.01e-10, 0.30e-10) if i == 0 else jnp.zeros_like(t)
    )
    for i in range(num_outputs)
}

time_sim.build_model(model_parameters=options, center_wvl=center_wvl, dt=dt, max_size=1)
modelResult = time_sim.run(t, inputs)

modelResult.plot_sim()
# plt.figure()
# plt.plot(t, gaussian, label='Gaussian Pulse')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Gaussian Pulse')
# plt.show()

# for i in range_of_phases:
#     f_mod =0
#     m = f_mod * jnp.ones(len(t),dtype = complex)
#     models['phase_modulator'] = Modulator(mod_signal=m)
#     time_sim = TimeSim(
#         netlist=netlist,
#         models=models,
#         # active_components=active_components,
#     )
#     time_sim.build_model(model_parameters=options, center_wvl=center_wvl, dt=dt, max_size= 3)

#     num_outputs = 2
#     inputs = {
#                 f'o{i}': smooth_rectangular_pulse(t, 0.01e-10, 0.30e-10) if i == 0 else jnp.zeros_like(t)
#                 for i in range(num_outputs)
#             }


#     modelResult =time_sim.run(t, inputs)


#     outputs = modelResult.outputs
#     phases_at_o1.append(outputs['o1'][-1])
#     # phases_at_o2.append(outputs['o2'][-1])
#     if i == -2*jnp.pi or i == 2*jnp.pi or i == 0:
#         modelResult.plot_sim()

#         # print(phases_at_o2[-1])

# phases = jnp.array(phases_at_o1)
# # phases2 = jnp.array(phases_at_o2)

# plt.figure()
# plt.plot(range_of_phases, jnp.abs(phases)**2, label='o1 last')
# # plt.plot(range_of_phases, jnp.abs(phases2)**2, label='o2 last')
# plt.xlabel('Applied phase (rad)')
# plt.ylabel('Final output signal')
# plt.legend()
# plt.show()
