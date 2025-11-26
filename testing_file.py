from functools import partial
import os
os.environ["OPENBLAS_NUM_THREADS"] = "12"

import sax
from simphony.libraries import siepic
from ctypes import Array
import numpy as np, matplotlib.pyplot as plt
from pyparsing import Union
from simphony.circuit import Circuit
from simphony.simulation import SampleModeSimulation
from simphony.simulation.sample_mode import SampleModeSimulationParameters, SampleModeSimulation
from simphony.circuit.components import SampleModeComponent
from simphony.signals import SampleModeOpticalSignal
import jax
import jax.numpy as jnp
from commpy.filters import rrcosfilter
from scipy.signal import upfirdn 
from scipy.integrate import solve_ivp
import sys
from pathlib import Path

gm_models_root = Path(__file__).resolve().parent / "Green-Machine" / "models"
sys.path.append(str(gm_models_root))
import gm_constructor as gmc

# def minmax(u):
#     return (u - u.min())/(u.max() - u.min())
# key = jax.random.PRNGKey(0)
# key2 = jax.random.PRNGKey(1)
# sigma, rho, beta = 10.0, 28.0, 8/3

# def lorenz(t, xyz):
#     x, y, z = xyz
#     return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

# N_coarse = N
# dtt = 0.05
# upsample = 1

# T = dtt * (N_coarse - 1)

# dt_fine = dtt / upsample
# t_fine = np.arange(0.0, T + 1e-12, dt_fine)

# sol_fine = solve_ivp(
#     lorenz, (0.0, T), (0., 1., 1.05),
#     method="RK45", max_step=dt_fine, t_eval=t_fine
# )
# x_f, y_f, z_f = sol_fine.y

# signal  = minmax(x_f)
# signal2 = minmax(y_f)
# y       = minmax(z_f)
# combined_sig = signal + signal2
# combined_sig = minmax(combined_sig)*jnp.pi
# def rrc_shape(symbols, h, sps):
#     x = upfirdn(h.astype(np.float32), np.asarray(symbols, np.float32), up=sps)
#     d = (len(h) - 1) // 2
#     return jnp.asarray(x[d:d + len(symbols)*sps], dtype=jnp.float32)
# sps = 50
# beta = 0.5
# span = 4
# _, h = rrcosfilter(span*sps+1, beta, 1.0, sps)
# phase_wave1 = rrc_shape(combined_sig,h, sps)

angles = ["_77.5", "_80", "_82.5", "_85", "_87.5", "_90"]
    # gap = "0.492"
    # gaps = [str(round(x, 3)) for x in np.arange(.476, 0.513, 0.004)]
    # deltas = [str(round(x, 3)) for x in np.arange(-0.038, -0.057, -0.002)]

    # delta = deltas[gaps.index(gap)]
config = gmc.configs[3] #3 is the index for the 89.4 wide sweep
gap = "0.45"
thicc = "263"
models = gmc.get_model(*config, thickness = thicc, gap = gap, angle = angles[3], folder = "./sparams/89.4 high mesh sweep", MA = 6)
netlist = gmc.gm_flat_netlist

def GM(wl: Union[float, Array] = 1.55):
    def load_sdict_npz(fname):
        d = np.load(fname)
        wls = d["wavelengths"]
        S = {}
        for k in d.files:
            if k == "wavelengths": continue
            # keys like "S_out3_in5"
            _, to_p, from_p = k.split("_", 2)
            S[(to_p, from_p)] = np.squeeze(d[k])
        return wls, S
    w_i, Sp = load_sdict_npz("sdict_exports/lumerical_bestfit_sdict_back_interp2000.npz")
    Sp_actual = {}
    for key in Sp:
        if key[0].startswith("in0") and key[1].startswith("out0"):
            Sp_actual[key] = Sp[key]
        if key[0].startswith("out0") and key[1].startswith("in0"):
            Sp_actual[key] = Sp[key]
        
    return Sp_actual


class StreamingCombSourceTime(SampleModeComponent):
    optical_ports = ["o0"]

    def __init__(self, base_wavelengths, scale=1.0):
        bw = jnp.asarray(base_wavelengths, jnp.float32).reshape(-1)
        self.base  = bw
        self.M     = int(bw.shape[0])
        self.scale = jnp.float32(scale)

    def sample_mode_initial_state(self, simulation_parameters):
        phi = jnp.zeros((self.M,), jnp.float32)
        t_idx = jnp.int32(0)
        return (phi, t_idx)

    @partial(jax.jit, static_argnums=(0,))
    def sample_mode_step(self, inputs, state, simulation_parameters):
        phi, t_idx = state
        A = (self.scale * jnp.exp(1j * phi).astype(jnp.complex64))[:, None]
        wls = self.base.astype(jnp.float32)
        out = {"o0": SampleModeOpticalSignal(amplitude=A, wavelength=wls)}
        return out, (phi, t_idx + jnp.int32(1))

# netlist = {
#     "instances": {
#         "laser1":"laser1",
#         "gm":"GM",
        
#     },
    
#     "connections": {
#         # "laser1,o0":"wg1,o0"
#         f"laser1,o0": f"gm,in0",
    
#     },
#     "ports": {
#         "o0": "gm,out0",
#         "o1": "gm,out1",
#         "o2": "gm,out2",
#         "o3": "gm,out3",
#         "o4": "gm,out4",
#         "o5": "gm,out5",
#         "o6": "gm,out6",
#         "o7": "gm,out7",
#     }
# }

# models = {
#     "laser1": StreamingCombSourceTime,
#     "GM": GM,
# }
netlist["instances"]["laser1"] = "laser1"
netlist["connections"][f"laser1,o0"] = f"edge_0__coupler0,o0"
netlist["ports"].pop("in0")

print(netlist["instances"])
models["laser1"] = StreamingCombSourceTime
wavelengths = jnp.linspace(1.50e-6, 1.60e-6, 50, jnp.float32)
dt = 1e-14
t = jnp.arange(0, 100000)*dt
print(len(t))
settings = {
    "laser1": {
    "base_wavelengths": wavelengths,
    "scale":jnp.sqrt(1.0),
    },
}
for key in netlist["instances"]:
    value = netlist["instances"][key]
    if value == "ewg":
        settings[key] = {"neff": 1.587838}
    if value == "wg":
        settings[key] = {"neff": 1.587838}
print(settings)

circuit = Circuit(netlist=netlist, models=models)
sim = SampleModeSimulation(circuit = circuit)
simulation_parameters = SampleModeSimulationParameters(
    optical_baseband_wavelengths=wavelengths,
    num_time_steps=int(len(t)), sampling_period=dt)

results50 = sim.run(simulation_parameters=simulation_parameters, settings=settings)
jax.block_until_ready(results50)
plt.plot(np.abs(results50["o2"]["output"].amplitude[:20000,0,0])**2)
plt.show()
points = {}
for j in range(8):
    points[f"o{j}"] = results50[f"o{j}"]["output"].amplitude[20000,:,0]
plt.plot(np.abs(points["o2"])**2)
plt.show()
np.savez(f"gm_output_signal_2_point.npz", points = points)
#20 minutes using just one max_order_model
