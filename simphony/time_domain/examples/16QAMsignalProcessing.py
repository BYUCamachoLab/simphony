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
from math import log2

import time

T = 4e-11
dt = 1e-14      # Total time duration (40 ps)
t = jnp.arange(0, T, dt)  # Time array
t0 = 1e-11  # Pulse start time
std = 1e-12

f_mod = 0
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

timePhaseInstantiated = Modulator(mod_signal=m)

netlist={
    "instances": {
        "wg": "waveguide",
        # "y": "y_branch",
        "pm": "phase_modulator",
        "pm2": "phase_modulator",
        
        # "y2": "y_branch",
        # "wg2": "waveguide",
        # "y3": "y_branch",
        # "y4": "y_branch",
        # "y5": "y_branch",
        # "y6": "y_branch",
        # "wg3": "waveguide",
        # "wg4": "waveguide",
        # "wg5": "waveguide",
        # "wg6": "waveguide",
        # "bdc": "bidirectional",
        # "bdc2": "bidirectional",
        # "bdc3": "bidirectional",
        # "bdc4": "bidirectional",
    },
    "connections": {
        # "bdc,port_2":"bdc2,port_1",
        # "bdc,port_4":"bdc2,port_3",

        # # "bdc2, port_2":"bdc3,port_1",
        
        # "pm2,o0":"bdc2, port_2",
        # "bdc3, port_1":"pm2,o1",

        # # "bdc2, port_4": "bdc3, port_3",

        # "pm,o0":"bdc2, port_4",
        # "bdc3, port_3":"pm,o1",

        # "bdc3, port_2": "bdc4,port_1",
        # "bdc3, port_4": "bdc4,port_3",
        "wg,o1":"pm,o0",

    },
    "ports": {
        # "o0": "bdc,port_1",
        # "o1": "bdc,port_3",
        # "o2": "bdc4, port_2",
        # "o3": "bdc4, port_4",
        "o0": "wg,o0",
        "o1": "pm,o1",
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
num_symbols = 10   # For example, 40 symbols so that 40*100 = 4000 time steps.
hold_time = 200  # Hold each symbol for 70 time steps.


def generate_16qam_piecewise_linear_signal(T, dt, num_symbols, hold_time):
    """
    Generate a 16-QAM signal with piecewise linear transitions where the hold time
    is specified and the ramp time is calculated so that the overall number of time steps is maintained.
    
    Parameters:
      T         : Total duration (s)
      dt        : Time step (s)
      num_symbols: Number of symbols
      hold_time : Desired hold time (number of time steps per symbol)
      
    Returns:
      t              : Time vector (s)
      I_waveform     : In-phase component (length = total time steps)
      Q_waveform     : Quadrature component (length = total time steps)
      signal_complex : Complex baseband signal = I_waveform + j Q_waveform
    """
    total_time_steps = int(T/dt)
    samples_per_symbol = total_time_steps // num_symbols  # e.g., 4000/40 = 100
    # Calculate ramp_time automatically:
    ramp_time = samples_per_symbol - hold_time
    if ramp_time < 0:
        raise ValueError("Hold time is too large; must be less than samples_per_symbol.")

    # Use np.linspace to get exactly the expected number of time steps.
    t = np.linspace(0, T, num_symbols * samples_per_symbol, endpoint=False)
    
    # 16-QAM: 4 bits per symbol.
    total_bits = num_symbols * 4
    data_bits = np.random.randint(0, 2, total_bits)
    
    # Mapping function for PAM4 (for both I and Q):
    # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    def bits_to_level(b0, b1):
        if (b0, b1) == (0, 0):
            return 1
        elif (b0, b1) == (0, 1):
            return 3
        elif (b0, b1) == (1, 1):
            return 5
        elif (b0, b1) == (1, 0):
            return 7

    symbols_I = []
    symbols_Q = []
    for i in range(0, total_bits, 4):
        bI0, bI1, bQ0, bQ1 = data_bits[i:i+4]
        symbols_I.append(bits_to_level(bI0, bI1))
        symbols_Q.append(bits_to_level(bQ0, bQ1))
    symbols_I = np.array(symbols_I)
    symbols_Q = np.array(symbols_Q)
    
    def piecewise_linear_hold(symbols, hold_time, ramp_time):
        """
        For each pair of symbols, hold the symbol value for 'hold_time' steps,
        then ramp linearly to the next symbol over 'ramp_time' steps.
        For the final symbol, hold it for the full period.
        """
        output = []
        for i in range(len(symbols) - 1):
            hold = np.full(hold_time, symbols[i])
            ramp = np.linspace(symbols[i], symbols[i+1], ramp_time, endpoint=False)
            block = np.concatenate([hold, ramp])
            output.append(block)
        # Last symbol: hold for full period.
        last_block = np.full(hold_time + ramp_time, symbols[-1])
        output.append(last_block)
        return np.concatenate(output)
    
    I_waveform = piecewise_linear_hold(symbols_I, hold_time, ramp_time)
    Q_waveform = piecewise_linear_hold(symbols_Q, hold_time, ramp_time)
    signal_complex = I_waveform + 1j * Q_waveform
    
    return t, I_waveform, Q_waveform, signal_complex
t, I_waveform, Q_waveform, signal_complex = generate_16qam_piecewise_linear_signal(T, dt, num_symbols, hold_time)

num_outputs = 2    
inputs = {
    f'o{i}': signal_complex if i == 0 else jnp.zeros_like(t)
    for i in range(num_outputs)
}

plt.plot(t, jnp.abs(signal_complex)**2 )
# inputs = {
#     f'o{i}': gaussian_pulse(t, t0 - 0.5 * t0, std) if i == 0 else jnp.zeros_like(t)
#     for i in range(num_outputs)
# }
# inputs = {
#             f'o{i}': smooth_rectangular_pulse(t,0.5e-11,2.5e-11) if i == 0 else jnp.zeros_like(t)
#             for i in range(num_outputs)
#         }


tic = time.time()
modelResult =time_sim.run(t, inputs)
toc = time.time()
run_time = toc - tic

print(f"Build time: {build_time}")
print(f"Run time: {run_time}")

plot_time_result(modelResult)
I_original = np.real(signal_complex)
Q_original = np.imag(signal_complex)
I_output = jnp.abs(np.real(modelResult.outputs["o1"]))
Q_output = jnp.abs(np.imag(modelResult.outputs["o1"]))

plt.figure(figsize=(10,8))

# Plot I components
plt.subplot(2,1,1)
plt.plot(t*1e12, I_original, label='Original I(t)', color='blue')
plt.plot(t*1e12, Q_output, label='System Output I(t)', color='green', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.title('In-Phase Component Comparison')
plt.legend()
plt.grid(True)

# Plot Q components
plt.subplot(2,1,2)
plt.plot(t*1e12, Q_original, label='Original Q(t)', color='red')
plt.plot(t*1e12, I_output, label='System Output Q(t)', color='orange', linestyle='--')
plt.xlabel('Time (ps)')
plt.ylabel('Amplitude')
plt.title('Quadrature Component Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.plot(I_original, Q_original, color='blue', linewidth=1, alpha=0.7, label='Transition Path')
#plt.scatter(symbols_I, symbols_Q, color='red', s=50, zorder=5, label='Symbols')
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation with Transition Paths")
plt.legend()
plt.grid(True)
plt.axis('equal')
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




