import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)
from scipy import signal
from simphony.utils import dict_to_matrix
from simphony.time_domain.simulation import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
from simphony.libraries import ideal

import time

T = 4e-11
dt = 1e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt) # Time array
t0 = 1.0e-11  # Pulse start time
std = 1e-12


f_mod =0
m = f_mod * jnp.ones(len(t),dtype = complex)
f_mod2 = 7.5*jnp.pi/8
m2 = f_mod2 * jnp.ones(len(t),dtype = complex)


timePhaseInstantiated = Modulator(mod_signal=m2)

netlist={
    "instances": {
        "dc":"directional_coupler",
        "pm":"phase_modulator",
        "wg":"waveguide",
        "dc2":"directional_coupler",
        "t":"terminator",
        "wg2":"waveguide",
        
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
        "o0":"dc, port_2",
        "o1":"dc, port_4",
        "o2":"dc2, port_4",
        
    },
}
models={
    "waveguide": siepic.waveguide,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
    "directional_coupler": siepic.directional_coupler,
    "terminator": siepic.terminator,
}
active_components = {
    "pm"
}

num_measurements = 200
model_order = 50
center_wvl = 1.54
wvl = np.linspace(1.50, 1.60, num_measurements)
options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100},'dr':{"coupling_length": 10},'dr2':{"coupling_length": 12.5} }

# time_sim = TimeSim(
#     netlist=netlist,
#     models=models,
#     active_components=active_components,
# )
# num_outputs = 3
# inputs = {
#                 f'o{i}': smooth_rectangular_pulse(t, 0.01e-10, T) if i == 0 else jnp.zeros_like(t)
#                 for i in range(num_outputs)
#             }

# time_sim.build_model(model_parameters=options, center_wvl=center_wvl, dt=dt, max_size= 3)
# modelResult =time_sim.run(t, inputs)

# modelResult.plot_sim()

range_of_phases = jnp.linspace(0, 2*jnp.pi, 100)
center_wvl_array = [1.52,1.53,1.54,1.55,1.56,1.57,1.58]
best_phase_list = {}
highest_result_list= {}
for j in center_wvl_array:
    best_phase = 0
    highest_result = 0
    for i in range_of_phases:
        f_mod =i
        m = f_mod * jnp.ones(len(t),dtype = complex)
        models['phase_modulator'] = Modulator(mod_signal=m)
        time_sim = TimeSim(
            netlist=netlist,
            models=models,
            active_components=active_components,
        )
        time_sim.build_model(model_parameters=options, center_wvl=j, dt=dt, max_size= 3)

        num_outputs = 3
        inputs = {
                    f'o{i}': smooth_rectangular_pulse(t, 0.01e-10, T) if i == 0 else jnp.zeros_like(t)
                    for i in range(num_outputs)
                }

        

        modelResult =time_sim.run(t, inputs)

        
        modelResult.plot_sim()
        outputs = modelResult.outputs["o2"]

        if jnp.abs(outputs[-100])**2 > highest_result:
            highest_result = jnp.abs(outputs[-100])**2
            best_phase = i
    best_phase_list[j] = best_phase
    highest_result_list[j] = highest_result
    
        
        



# range_of_phases = jnp.linspace(-2*jnp.pi, 2*jnp.pi, 50)
# phases_at_o1 = []
# phases_at_o2 = []

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
#     phases_at_o2.append(outputs['o2'][-1])
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




