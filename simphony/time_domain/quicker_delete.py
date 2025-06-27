import time
import numpy as np

import jax
# jax.config.update("jax_log_compiles", True)  # Use double precision
from jax import jit
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "simphony")))
import simphony
from simphony.time_domain import TimeSim
from simphony.time_domain.utils import gaussian_pulse, smooth_rectangular_pulse 
from simphony.libraries import siepic, ideal
from simphony.time_domain.ideal import Modulator,MMI
import sax
import jax.numpy as jnp
from simphony.time_domain.time_system import (
    BlockModeSystem,
    SampleModeSystem,
    TimeSystem,
    TimeSystemIIR,
)
from simphony.time_domain.pole_residue_model import BVF_Options, IIRModelBaseband
from simphony.utils import dict_to_matrix
# ── your original step function ───────────────────────────────────────────────
netlist = {
    "instances":{
        "wg1": "waveguide",
        "wg2": "waveguide",
        
        "yb1": "y_branch",
        "yb2": "y_branch",
    },
    "connections":{
        "wg1,o0":"yb1,port_2",
        "wg2,o0":"yb1,port_3",

        "wg2,o1":"yb2,port_2",
        "wg1,o1":"yb2,port_3",


    },
    "ports":{
        "o0":"yb1,port_1",
        "o1":"yb2,port_1",

    },
}
T = 100e-11
dt = 1e-14                   # Time step/resolution
t = jnp.arange(0, T, dt)
num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)
options = {
    'wl': wvl,
    'wg1': {'length': 10.0},
    'wg2': {'length': 50.0},
    'wg3': {'length': 50.0},
    'wg4': {'length': 50.0},
    'wg5': {'length': 50.0},
    'wg6': {'length': 50.0},
    'wg7': {'length': 50.0},
    'wg8': {'length': 50.0},
    'wg9': {'length': 50.0},
    'wg10': {'length': 50.0},
    
}
models = {
    "waveguide": siepic.waveguide,
    "y_branch": siepic.y_branch,
}

inputs = {
    "o0":smooth_rectangular_pulse(t, 0.0, T+ 20.0e-11),
    'o1': jnp.zeros_like(t),
    }

ports = sorted(inputs.keys(), key=lambda k: int(k[1:]))  
signals = [ inputs[p] for p in ports ]   
u = jnp.stack(signals, axis=1)         


inputs_per_t = tuple(
    tuple(u[t].tolist())                  
    for t in range(u.shape[0])
)

circuit, _ = sax.circuit(
                            netlist=netlist,
                            models=models,
                        )

s_params_dict = circuit(**options)
s_matrix = np.asarray(dict_to_matrix(s_params_dict))
center_wvl = 1.55
c_light = 299792458
center_freq = c_light / (center_wvl * 1e-6)
freqs = c_light / (wvl * 1e-6) - center_freq
sampling_freq = -1 / dt
beta = sampling_freq / (freqs[-1] - freqs[0])
bvf_options = BVF_Options(beta=beta)
sorted_ports = sorted(netlist["ports"].keys(), key=lambda p: int(p.lstrip('o')))

iir_model = IIRModelBaseband(
    wvl, center_wvl, s_matrix,order = 50, options=bvf_options
)
iir_model2 = IIRModelBaseband(
    wvl, center_wvl, s_matrix,order = 50, options=bvf_options
)
iir_model3 = IIRModelBaseband(
    wvl, center_wvl, s_matrix,order = 50, options=bvf_options
)
iir_model4 = IIRModelBaseband(
    wvl, center_wvl, s_matrix,order = 50, options=bvf_options
)

N_STEPS = len(t)
td1 = TimeSystemIIR(iir_model, sorted_ports)
td2 = TimeSystemIIR(iir_model2, sorted_ports)
td3 = TimeSystemIIR(iir_model3, sorted_ports)
td4 = TimeSystemIIR(iir_model4, sorted_ports)
systems = (td1, td2, td3, td4)
initial_state1 = td1.init_state()
initial_state2 = td2.init_state()
initial_state3 = td3.init_state()
initial_state4 = td4.init_state()
n_runs = len(t)

jitted_step1 = td1.step
jitted_step2 = td2.step
jitted_step3 = td3.step
jitted_step4 = td4.step

x_warm1, y_warm1 = jitted_step1(initial_state1, inputs_per_t[0])
x_warm2, y_warm2 = jitted_step2(initial_state2, inputs_per_t[0])
x_warm3, y_warm3 = jitted_step3(initial_state3, inputs_per_t[0])
x_warm4, y_warm4 = jitted_step4(initial_state4, inputs_per_t[0])
x_warm1.block_until_ready()
x_warm2.block_until_ready()
x_warm3.block_until_ready()
x_warm4.block_until_ready()
_ =    td1.step(initial_state1, inputs_per_t[0])
_ =    td2.step(initial_state2, inputs_per_t[0])
_ =    td3.step(initial_state3, inputs_per_t[0])
_ =    td4.step(initial_state4, inputs_per_t[0])

# Your connection definition
connections = [
    (("port_in", "o0"), ("td1", "o0")),
    (("td1", "o1"), ("td2", "o0")),
    (("td2", "o1"), ("td3", "o0")),
    (("td3", "o1"), ("td4", "o0")),
    (("td4", "o1"), ("port_out", "o1")),
]

# Create a unique set of all (system, port) pairs
nodes = set()
for a, b in connections:
    nodes.add(a)
    nodes.add(b)

# Initialize a table to hold values at each port/node
# You can initialize to None, zeros, or whatever is appropriate
table = {node: None for node in nodes}

# Example: Put a value at port_in o0
table[("port_in", "o0")] = 42

# Pass values along the connections
for a, b in connections:
    # For demonstration, pass value from a to b
    table[b] = table[a]
    # You could add processing, e.g., apply a function

inner_states = {
    "td1": initial_state1,
    "td2": initial_state2,
    "td3": initial_state3,
    "td4": initial_state4,
}
systems = {
    "td1": td1,
    "td2": td2,
    "td3": td3,
    "td4": td4,
}
# All unique system/port pairs
system_port_labels = {}
for sys_name, sys_obj in systems.items():
    # Assuming sys_obj has .outputs or just 2 outputs (o0, o1) for simplicity
    system_port_labels[sys_name] = [f"o{i}" for i in range(sys_obj.sys.D.shape[0])]
    # Or hardcode: ["o0", "o1"] if always two outputs

# The value table
table = {}
# Bi-directional lookup
bi_lookup = {}
for a, b in connections:
    bi_lookup[a] = b
    bi_lookup[b] = a
jit_steppers = {
    "td1": jitted_step1,
    "td2": jitted_step2,
    "td3": jitted_step3,
    "td4": jitted_step4,
}

# Initialization (above)
nodes = set()
for a, b in connections:
    nodes.add(a)
    nodes.add(b)
table = {node: 0 for node in nodes}

# Simulation loop
x = initial_state1
t0 = time.perf_counter()
for step, inp in enumerate(inputs_per_t):
    # Update input port for current step
    table[('port_in', 'o0')] = inp[0]  # or use the correct input index

    for key, system in systems.items():
        x, out_tuple = jit_steppers[key](inner_states[key], inp)
        inner_states[key] = x
        x.block_until_ready()
        for port_label, out_val in zip(system_port_labels[key], out_tuple):
            table[(key, port_label)] = out_val

        
    
    for src, dst in connections:
        table[dst] = table[src]
    
    # Optionally accumulate or print lookup times here

t1 = time.perf_counter()
dispatch_plus_compute = (t1 - t0) / n_runs

microseconds = dispatch_plus_compute * 1e6
print(f"Total time per run: {microseconds:.6f} μs")

# ── 2) Dispatch + Compute ───────────────────────────────────────────────────
x = initial_state1
t0 = time.perf_counter()
for inp in inputs_per_t:
    x, _ = jitted_step1(x, inp)
    x.block_until_ready()
t1 = time.perf_counter()
dispatch_plus_compute = (t1 - t0) / n_runs

print(f"avg jitted loop (per-step)     : {dispatch_plus_compute*1e6:7.2f} µs")