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
for i in range(4):

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
        w_i, Sp = load_sdict_npz("gm_real_sdict_1p50_1p60um_1000pts.npz")
        return Sp


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
        
    class OpticalModulatorTime(SampleModeComponent):
        optical_ports = ["o0", "o1"]

        def __init__(self, *, length=1.0, operating_wl=1.55e-6, effective_index=2.4,
                    group_index=4.2, mod_signal=0.0):
            self.length = float(length)
            self.operating_wl = float(operating_wl)
            self.effective_index = float(effective_index)
            self.group_index = float(group_index)
            self.mod_signal = jnp.asarray(mod_signal)

        def sample_mode_initial_state(self, simulation_parameters):
            return jnp.int32(0)

        def sample_mode_step(self, inputs, state, simulation_parameters):
            a0 = inputs["o0"].amplitude  # forward field (W,M)
            a1 = inputs["o1"].amplitude  # reverse field (if any)
            lam = inputs["o0"].wavelength

            if a0.ndim == 1: a0 = a0[:, None]
            if a1.ndim == 1: a1 = a1[:, None]
            dtype = a0.dtype
            

            phase_t = self.mod_signal[state]
            

            coeff = jnp.exp(1j * (phase_t)).astype(dtype)
            out0 = SampleModeOpticalSignal(jnp.zeros_like(a1), lam)
            out1 = SampleModeOpticalSignal(a0 * coeff, lam)

            return {"o0": out0, "o1": out1}, state + 1

    netlist = {
        "instances": {
            "laser1":"laser1",
            "gm":"GM",
            
        },
        
        "connections": {
            # "laser1,o0":"wg1,o0"
            f"laser1,o0": f"gm,in{i}",
        
        },
        "ports": {
            "o0": "gm,out0",
            "o1": "gm,out1",
            "o2": "gm,out2",
            "o3": "gm,out3",
            "o4": "gm,out4",
            "o5": "gm,out5",
            "o6": "gm,out6",
            "o7": "gm,out7",
        }
    }

    models = {
        "laser1": StreamingCombSourceTime,
        "GM": GM,
    }
    wavelengths = jnp.linspace(1.50e-6, 1.60e-6, 50, jnp.float32)
    dt = 1e-14
    t = jnp.arange(0, 100000)*dt
    print(len(t))
    settings = {
        "laser1": {
        "base_wavelengths": wavelengths,
        "scale":jnp.sqrt(1.0),
        },
        "gm":{
            "max_order":500
            },
    }

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
    np.savez(f"gm_output_signal_{i}_point.npz", points = points)
#20 minutes using just one max_order_model