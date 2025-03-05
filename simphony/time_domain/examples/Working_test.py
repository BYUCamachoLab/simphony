import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sax
from jax import config
config.update("jax_enable_x64", True)

from simphony.time_domain.TimeSim import TimeSim,TimeResult
from simphony.time_domain.utils import  gaussian_pulse, smooth_rectangular_pulse
from simphony.libraries import siepic
from simphony.time_domain.ideal import Modulator
import matplotlib

import time

T = 20.0e-11 
dt = 1e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt)  # Time array
t0 = 1e-11  # Pulse start time
std = 1e-12
inter = 50
m   = jnp.array([], dtype=jnp.complex128)  # MZI#1 arm A
m4  = jnp.array([], dtype=jnp.complex128)  # MZI#1 arm B
m2  = jnp.array([], dtype=jnp.complex128)  # MZI#2 arm A
m5  = jnp.array([], dtype=jnp.complex128)  # MZI#2 arm B
#np.pi/6,-np.pi/6,np.pi/3,-np.pi/3,
#np.pi/6,-np.pi/6,np.pi/3,-np.pi/3, 
i_phase_table = [ np.pi/10, 9*np.pi/10, np.pi/2.3, np.pi/1.7 ]
q_phase_table = [ np.pi/10, 9*np.pi/10, np.pi/2.3, np.pi/1.7 ]  

num_symbols = int(len(t)/inter) - 1

for sym in range(num_symbols):
    # 4 random bits:
    bits = np.random.randint(0,2,size=4)
    
    # 2 bits for I, 2 bits for Q
    i_index = bits[0]*2 + bits[1]   # from 00..11 => 0..3
    q_index = bits[2]*2 + bits[3]   # from 00..11 => 0..3

    i_val = i_phase_table[i_index]
    q_val = q_phase_table[q_index]
    
    # Create the push-pull waveforms for one symbol interval
    i_signal_block_A = jnp.ones(inter) * i_val
    i_signal_block_B = jnp.ones(inter) * (-i_val)

    q_signal_block_A = jnp.ones(inter) * q_val
    q_signal_block_B = jnp.ones(inter) * (-q_val)

    # Append
    m  = jnp.concatenate([m, i_signal_block_A])
    m4 = jnp.concatenate([m4, i_signal_block_B])
    m2 = jnp.concatenate([m2, q_signal_block_A])
    m5 = jnp.concatenate([m5, q_signal_block_B])


# f_mod =0
# # m = f_mod * jnp.ones(len(t),dtype = complex)
# f_mod2 =0
# m2 = f_mod2 * jnp.ones(len(t),dtype = complex)
f_mod3 =jnp.pi/2 
m3 = f_mod3 * jnp.ones(len(t),dtype = complex)
# f_mod4 =0.0 
# # m4 = f_mod4 * jnp.ones(len(t),dtype = complex)
# m5 = f_mod4 * jnp.ones(len(t),dtype = complex)
x = jnp.linspace(0, 3.14, len(t))

# Define Gaussian parameters
mu = 1.14  # center the Gaussian in the middle of the interval
sigma = 0.3     # adjust sigma for desired width

# Compute the Gaussian function
gaussian = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Optionally, normalize so the area under the curve is 1
gaussian = gaussian / jnp.trapezoid(gaussian, x)
zero = 0*x
timePhaseInstantiated = Modulator(mod_signal=m)
timePhaseInstantiated2 = Modulator(mod_signal=m2)
timePhaseInstantiated3 = Modulator(mod_signal=m3)
timePhaseInstantiated4 = Modulator(mod_signal=m4)
timePhaseInstantiated5 = Modulator(mod_signal=m5)

netlist={
    "instances": {
        "wg": "waveguide",
        "y": "y_branch",
        "pm": "phase_modulator",
        "pm2": "phase_modulator2",
        "pm3": "phase_modulator3",
        "pm4": "phase_modulator4",
        "pm5": "phase_modulator5",
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
        
        # "bdc,port_3": "y3,port_1",
        # "bdc,port_4": "y4,port_1",
        "y2,port_2":"y3,port_1",
        "y2,port_3":"y4,port_1",
        
        "y4,port_2":"wg5,o0",
        "y4,port_3":"wg6,o0",
        "wg5,o1":"pm,o0",
        "wg6,o1":"pm4,o0",

        "y5,port_3":"pm,o1",
        "y5,port_2":"pm4,o1",
        

        "y3,port_2":"wg,o0",
        "y3,port_3":"wg2,o0",
        "wg,o1":"pm2,o0",
        "wg2,o1":"pm5,o0",

        "y6,port_2":"pm2,o1",
        "y6,port_3":"pm5,o1",
        "y6,port_1":"pm3,o0",
        
        
        "y,port_3":"pm3,o1",
        # "y,port_3":"y6,port_1",
        "y,port_2":"y5,port_1",

        

        # "bdc,port_1":"pm,o1",

    },
    "ports": {
        "o0":"y2, port_1",
        # "o0":"bdc,port_1",
        # "o1":"bdc,port_2",
        "o1":"y, port_1",
        # "o1":"y6,port_1",
        # "o2":"y5,port_1",


        
        # "o0": 'wg,o0',
        # "o1": "wg2,o0",
        #"o2": "y,port_1",

        # "o0": "bdc,port_1",
        # "o1": "bdc,port_2",
        # "o2": "wg3,o1",
        # "o3": "wg4,o1",

        # "o0":"pm,o0",
        # "o1":"bdc,port_2",
        # "o2":"bdc,port_3",
        # "o3":"bdc,port_4",

    },
}
models={
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
    "bidirectional": siepic.bidirectional_coupler,
    "phase_modulator": timePhaseInstantiated,
    "phase_modulator2": timePhaseInstantiated2,
    "phase_modulator3": timePhaseInstantiated3,
    "phase_modulator4": timePhaseInstantiated4,
    "phase_modulator5": timePhaseInstantiated5,
}
active_components = {
    "pm","pm2","pm3","pm4","pm5",
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
options = {'wl':wvl,'wg':{"length": 10.0, "loss": 100}, 'wg2':{"length": 10.0, "loss": 100},'wg3':{"length":10.0, "loss":100},'wg4':{"length":10.0, "loss":100},
           'wg5':{"length":10.0, "loss":100},'wg6':{"length":10.0, "loss":100}}

tic = time.time()
time_sim.build_model(model_parameters=options)
toc = time.time()
build_time = toc - tic

num_outputs = 2


# inputs = {
#     f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0 or i == 1  else jnp.zeros_like(t)
#     for i in range(num_outputs)
# }
inputs = {
            f'o{i}': smooth_rectangular_pulse(t,0.0e-11,20e-11) if i == 0 else jnp.zeros_like(t)
            for i in range(num_outputs)
        }


tic = time.time()
modelResult =time_sim.run(t, inputs)
toc = time.time()
run_time = toc - tic

print(f"Build time: {build_time}")
print(f"Run time: {run_time}")
modelResult.plot_sim()

I_output = np.real(modelResult.outputs["o1"])
Q_output = np.imag(modelResult.outputs["o1"])


plt.subplot(2,1,1)
plt.plot(t*1e12, Q_output, label='System Output I(t)', color='green', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.title('In-Phase Component Comparison')
plt.legend()
plt.grid(True)

# Plot Q components
plt.subplot(2,1,2)

plt.plot(t*1e12, I_output, label='System Output Q(t)', color='orange', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.title('Quadrature Component Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



plt.figure(figsize=(6,6))
plt.plot(Q_output, I_output, color='blue', linewidth=1, alpha=0.7, label='Transition Path')
#plt.scatter(symbols_I, symbols_Q, color='red', s=50, zorder=5, label='Symbols')
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation with Transition Paths")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

def upsample_trajectory(I, Q, factor=20):
    I_list, Q_list = [], []
    n = len(I)
    for i in range(n - 1):
        i0, i1 = I[i], I[i+1]
        q0, q1 = Q[i], Q[i+1]
        for alpha in np.linspace(0, 1, factor, endpoint=False):
            I_list.append(i0 + alpha*(i1 - i0))
            Q_list.append(q0 + alpha*(q1 - q0))
    # Add the last point
    I_list.append(I[-1])
    Q_list.append(Q[-1])
    return np.array(I_list), np.array(Q_list)
    
I_output,Q_output = upsample_trajectory(I_output,Q_output,factor=5)
plt.figure(figsize=(8,6))
bins = 500
plt.hist2d(I_output, Q_output, bins=bins, cmap='jet',norm=matplotlib.colors.LogNorm() )
plt.colorbar(label="Counts per bin")
plt.title("Heat Map of 16-QAM Trajectory (fmod = 0) with Upsampling" )
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.show()


# I_output1 = np.real(modelResult.outputs["o2"])
# Q_output1 = np.imag(modelResult.outputs["o2"])

# plt.subplot(2,1,1)
# plt.plot(t*1e12, Q_output1, label='System Output I(t)', color='green', linestyle='--')
# plt.xlabel('Time (ps)')
# plt.ylabel('Amplitude')
# plt.title('In-Phase Component Comparison')
# plt.legend()
# plt.grid(True)

# # Plot Q components
# plt.subplot(2,1,2)

# plt.plot(t*1e12, I_output1, label='System Output Q(t)', color='orange', linestyle='--')
# plt.xlabel('Time (ps)')
# plt.ylabel('Amplitude')
# plt.title('Quadrature Component Comparison')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,6))
# plt.plot(Q_output1, I_output1, color='blue', linewidth=1, alpha=0.7, label='Transition Path')
# #plt.scatter(symbols_I, symbols_Q, color='red', s=50, zorder=5, label='Symbols')
# plt.xlabel("In-Phase (I)")
# plt.ylabel("Quadrature (Q)")
# plt.title("16-QAM Constellation with Transition Paths")
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# # plt.plot(t, m)
# # plt.plot(t, m2)
# # plt.show()


