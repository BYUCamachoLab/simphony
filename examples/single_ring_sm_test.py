"""single_ring_sm_test.py.

Minimal test: sample-mode simulation of a single all-pass SiEPIC ring resonator
versus the exact S-parameter frequency response.

Goal: verify that the frequency-shifting code (exp(j*delta_omega)*A, B) drives the
sample-mode steady state to the correct S-parameter transmission at each wavelength.

Topology
--------
  OpticalCombSource ──► half_ring ──► [output]
                           │    ▲
                           ▼    │
                        waveguide (ring cavity)

Run
---
  cd <repo root>
  python examples/single_ring_sm_test.py
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from simphony.circuit.circuit import Circuit
from simphony.simulation.s_parameter import (
    SParameterSimulation,
    SParameterSimulationParameters,
)
from simphony.simulation.sample_mode import (
    SampleModeSimulation,
    SampleModeSimulationParameters,
)
from simphony.libraries.ideal.sources import OpticalCombSource, VoltageSource
from simphony.libraries.ideal.modulators import OpticalModulator
from simphony.libraries.siepic import half_ring, waveguide

# ── Ring parameters ───────────────────────────────────────────────────────────
# half_ring: gap (nm), radius (µm), width (nm), thickness (nm)
# waveguide:  length (µm), width (nm), height (nm), loss (dB/cm)
# Ring round-trip: π*radius + length ≈ π*5 + 5 ≈ 20.7 µm
# T_rt ≈ ng * L_rt / c ≈ 3.7 * 20.7e-6 / 3e8 ≈ 255 fs  →  dt must be well below this
HR = dict(pol="te", gap=100, radius=5, width=500, thickness=220, coupling_length=0)
WG = dict(pol="te", length=5.0, width=500, height=220, loss=100.0)

DT = 1e-14  # 10 fs  (dt ≪ T_rt ✓, and waveguide VF fit is near machine precision)
N_STEPS = 2000
TRANSIENT = 300  # steps to discard when averaging steady state

# ── Wavelength grids ──────────────────────────────────────────────────────────
wl_sp_um = np.linspace(1.50, 1.60, 1001)  # dense S-param grid (µm)
sm_wl_m = jnp.linspace(1.50, 1.60, 81) * 1e-6  # sample-mode probe wavelengths (m)
sm_wl_um = np.array(sm_wl_m) * 1e6  # same, in µm for plotting

# ── Vector-fitting parameters (passed per-component in pre-wrapped format) ────
VF = dict(
    model_order=None,  # auto-select via golden-section search
    min_model_order=2,
    max_model_order=30,
    num_frequency_samples=800,
    center_wavelength=1.55e-6,
    spectral_range=(1.50e-6, 1.60e-6),
)

# ── S-parameter simulation (ground truth) ─────────────────────────────────────
ring_netlist = {
    "instances": {"hr": "half_ring", "wg": "waveguide"},
    "connections": {"hr,port_2": "wg,o0", "wg,o1": "hr,port_4"},
    "ports": {"in": "hr,port_1", "out": "hr,port_3"},
}
sp_circuit = Circuit(ring_netlist, {"half_ring": half_ring, "waveguide": waveguide})
sp_settings = {"hr": HR, "wg": WG}
sp_sim = SParameterSimulation(sp_circuit, sp_settings, SParameterSimulationParameters())
sp_result = sp_sim.run(wl=wl_sp_um)
sp_T = np.abs(np.array(sp_result.s_parameters[("in", "out")])) ** 2

# S-param at exactly the sample-mode probe wavelengths (for direct comparison)
sp_at_sm = np.abs(np.array(sp_sim.run(wl=sm_wl_um).s_parameters[("in", "out")])) ** 2
print(f"S-parameter done.  T range: [{sp_T.min():.4f}, {sp_T.max():.4f}]")

# ── Sample-mode simulation ────────────────────────────────────────────────────
# Settings in pre-wrapped format so vector_fitting_parameters is picked up
# correctly alongside sax_settings (avoids the auto-wrap that would bury VF params).
MOD = dict(
    phase_coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
    absorption_coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
    length=1.0,
)
VS = dict(steady_state_voltage=0.0)

sm_netlist = {
    "instances": {
        "hr": "half_ring",
        "mod": "modulator",
        "wg": "waveguide",
        "vs": "voltage_source",
        "source": "comb_source",
    },
    "connections": {
        "hr,port_2": "mod,o0",
        "mod,o1": "wg,o0",
        "wg,o1": "hr,port_4",
        "vs,e0": "mod,e0",
        "source,o0": "hr,port_1",
    },
    "ports": {"out": "hr,port_3"},
}
sm_models = {
    "half_ring": half_ring,
    "waveguide": waveguide,
    "modulator": OpticalModulator,
    "voltage_source": VoltageSource,
    "comb_source": OpticalCombSource,
}
sm_circuit = Circuit(sm_netlist, sm_models)

sm_settings = {
    "hr": {"sax_settings": HR, "vector_fitting_parameters": VF},
    "wg": {"sax_settings": WG, "vector_fitting_parameters": VF},
    "mod": MOD,
    "vs": VS,
    "source": {"wavelength": sm_wl_m, "linewidth": 0.0},
}
sm_params = SampleModeSimulationParameters(
    optical_baseband_wavelengths=sm_wl_m,
    dt=DT,
    num_time_steps=N_STEPS,
)

sm_sim = SampleModeSimulation(
    sm_circuit,
    sm_settings,
    tracked_ports={"out": "hr,port_3"},
    simulation_parameters=sm_params,
)

print(
    f"Running sample-mode: {N_STEPS} steps × {len(sm_wl_m)} wavelengths, dt={DT:.0e} s …"
)
sm_result = sm_sim.run(use_jit=True)
print("Done.")

# ── Inspect tracked port signals ─────────────────────────────────────────────
print("\nTracked output signals:")
for port_name, sig in sm_result.output_signals.items():
    if hasattr(sig, "amplitude"):
        amp = np.array(sig.amplitude)
        print(
            f"  {port_name!r:20s}  shape={str(amp.shape):15s}  max|A|={np.abs(amp).max():.4f}"
        )

print("\nTracked input signals:")
for port_name, sig in sm_result.input_signals.items():
    if hasattr(sig, "amplitude"):
        amp = np.array(sig.amplitude)
        print(
            f"  {port_name!r:20s}  shape={str(amp.shape):15s}  max|A|={np.abs(amp).max():.4f}"
        )

# ── Extract the through-port output ──────────────────────────────────────────
out_sig = sm_result.output_signals["out"]
out_amp = np.array(out_sig.amplitude)  # (N_STEPS, L, M)
print(f"\nUsing output signal: 'out'  shape={out_amp.shape}")

# Steady-state: time-average of |amplitude|² after the transient
sm_T_steady = np.mean(np.abs(out_amp[TRANSIENT:, :, 0]) ** 2, axis=0)  # (L,)

# ── Comparison table ──────────────────────────────────────────────────────────
n_wl = len(sm_wl_m)
print(f"\n{'Wavelength (µm)':<18} {'S-param T':>10} {'SM steady T':>12} {'ratio':>8}")
print("-" * 52)
for i in range(0, n_wl, max(1, n_wl // 10)):
    wl = sm_wl_um[i]
    spt = sp_at_sm[i]
    smt = sm_T_steady[i]
    ratio = smt / spt if spt > 1e-6 else float("nan")
    print(f"  {wl:.4f}            {spt:>10.4f}   {smt:>10.4f}   {ratio:>8.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
t_ps = np.arange(N_STEPS) * DT * 1e12
res_i = int(np.argmax(sp_at_sm))  # wavelength index closest to a resonance

fig, axes = plt.subplots(3, 1, figsize=(10, 11))

# 1. Filter spectrum: S-param vs sample-mode steady state
ax = axes[0]
ax.plot(wl_sp_um, sp_T, "b-", lw=1.5, label="S-parameter (dense)")
ax.scatter(
    sm_wl_um,
    sm_T_steady,
    s=60,
    color="tomato",
    zorder=5,
    label=f"Sample-mode steady state (last {N_STEPS-TRANSIENT} steps)",
)
ax.plot(sm_wl_um, sm_T_steady, "r--", lw=1.0, alpha=0.6)
ax.set_ylabel("Through-port transmission")
ax.set_xlabel("Wavelength (µm)")
ax.set_title("All-Pass Ring: S-param vs Sample-Mode Steady State")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Transient at the resonant wavelength
ax = axes[1]
power_vs_t = np.abs(out_amp[:, res_i, 0]) ** 2
ax.plot(t_ps, power_vs_t, lw=1.2, color="steelblue", label="Sample-mode |A(t)|²")
ax.axhline(
    sp_at_sm[res_i],
    color="tomato",
    lw=1.5,
    ls="--",
    label=f"S-param T = {sp_at_sm[res_i]:.4f}",
)
ax.axvline(
    TRANSIENT * DT * 1e12, color="grey", ls=":", lw=1.0, label="Transient cutoff"
)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Power")
ax.set_title(f"Transient at resonance  λ ≈ {sm_wl_um[res_i]:.4f} µm")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter: SM steady T vs S-param T for all probe wavelengths
ax = axes[2]
ax.scatter(sp_at_sm, sm_T_steady, s=40, color="steelblue")
lim = max(sp_at_sm.max(), sm_T_steady.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="Perfect agreement")
ax.set_xlabel("S-parameter T")
ax.set_ylabel("Sample-mode steady-state T")
ax.set_title("Sample-mode vs S-param agreement across all wavelengths")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "examples/single_ring_sm_test.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved {out_path}")
