import numpy as np
import matplotlib.pyplot as plt
import sax
import random
import time

from simphony.utils import dict_to_matrix
from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal, siepic

from simphony.baseband_vector_fitting import Baseband_Model_SingleIO, BVF_Options, CVF_Options, CVF_Model
from scipy.signal import  StateSpace, dlsim, lsim
from simphony.quantum import QuantumState, CoherentState, SqueezedState, compose_qstate

# netlist = {
#     "instances": {
#         "wg": "waveguide",
#         "hr": "half_ring",
#     },
#     "connections": {
#         "hr,port 1": "wg,o0",
#         "hr,port 3": "wg,o1",
#     },
#     "ports": {
#         "o0": "hr,port 4",
#         "o1": "hr,port 2",
#     }
# }

# circuit, info = sax.circuit(
#     netlist=netlist,
#     models={
#         "waveguide": ideal.waveguide,
#         "half_ring": siepic.bidirectional_coupler,
#     }
# )

netlist = {
    "instances": {
        "wg": "waveguide",
    },
    "connections": {
    },
    "ports": {
        "o0": "wg,o0",
        "o1": "wg,o1",
    }
}

circuit, info = sax.circuit(
    netlist=netlist,
    models={
        "waveguide": ideal.waveguide,
    }
)

num_measurements = 100
# model_order = np.arange(50, 60, 1)
model_order = 100
wvl = np.linspace(1.5, 1.6, num_measurements)

center_wvl = 1.5493
# center_wvl = 1.58
neff = 2.34
ng = 3.4
c = 299792458
ring_length = 20.0
stepping_time = ring_length * 1e-6 / (c / ng)

S = circuit(wl=wvl, wg={"length": ring_length, "loss": 100})

s_params = dict_to_matrix(S)
plt.plot(np.abs(s_params[:, 1, 0])**2)
plt.show()


kwargs = (wvl, {"length": ring_length, "loss": 100})
options = CVF_Options(quantum=True,pole_spacing='log',alpha=0.01, beta=1.0, baseband=True, enforce_stability=True, poles_estimation_threshold=100, max_iterations=10, model_error_threshold = 1e-10, dt=1e-15, order=model_order, center_wvl=center_wvl)

# qte = QuantumTimeElement(circuit, options, wl=wvl, wg={"length": ring_length, "loss": 100})
ring_resonator = CVF_Model(wvl, S, options)
ring_resonator.plot_frequency_domain_model()

# in_state = CoherentState("o0", 1.0 + 0.0j)
in_state = SqueezedState("o0", r=1.0, phi=0, alpha=5.0 + 5.0j)
vac = CoherentState("o1", 0.0 + 0.0j)

qstate = compose_qstate(in_state, vac)
qstate._add_vacuums(2)
qstate.to_xxpp()

N = int(10000)
input_states = [qstate for i in range(N)]

T = 200.0e-12

t = np.linspace(0, T, N)
sig = np.exp(1j*2*np.pi*t*0).reshape(-1, 1)
sig = np.hstack([sig, np.zeros_like(sig)])

xout_means = None
xout_cov = None
means = np.array([])
cov = np.array([])
new_qstates = []
count = 0
for in_state in input_states:
    print(count)
    yout_means, yout_cov, xout_means, xout_cov = ring_resonator.state_space.qstep(in_state, x0_means=xout_means, x0_cov=xout_cov)
    if len(means) == 0:
        means = yout_means
    else:
        means = np.vstack([means, yout_means])

    if len(cov) == 0:
        cov = yout_cov[np.newaxis,:,:]
    else:
        cov = np.concatenate([cov, yout_cov[np.newaxis, :, :]], axis=0)

    # new_qstates.append(QuantumState(means, cov, qstate.ports, convention="xxpp"))
    if count == 50:
        count = 0
        new_qstates.append(QuantumState(yout_means, yout_cov, convention="xxpp"))
    else:
        count += 1
    

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# new_qstates[0].plot_mode(mode=0, ax=axes[0])
# new_qstates[0].plot_mode(mode=1, ax=axes[1])

plt.ion()
angles = np.array([])
for qs in new_qstates:

    x = qs.means[1]
    p = qs.means[5]
    angle = np.angle(x + 1j*p)
    angles = np.append(angles, angle)
    print(f"x: {x} + p: {p}")
    print(f"angle: {angle}")

    qs.plot_mode(mode=1)
    # plt.show()
    plt.pause(0.01)
    # plt.clf()
    plt.draw()




plt.plot(np.abs(means[:, 1] + 1j*means[:, 5])**2)
plt.plot(np.abs(cov[:, 0,0] + 1j*cov[:, 4, 4]))
plt.plot(np.abs(cov[:, 1,1] + 1j*cov[:, 5, 5]))

plt.show()

plt.plot(cov[:, 1])
plt.show()

plt.plot(np.abs(y)**2)
plt.show()


print(f"{ring_resonator.error}")
plt.scatter(ring_resonator.initial_poles().real, ring_resonator.initial_poles().imag)
plt.scatter(ring_resonator.poles.real, ring_resonator.poles.imag)
plt.show()
# ring_resonator.plot_frequency_domain_model()
ring_resonator.plot_frequency_domain_model()
ring_resonator.compute_steady_state()

N = int(15000)
T = 200.0e-12

t = np.linspace(0, T, N)
sig = np.exp(1j*2*np.pi*t*0).reshape(-1, 1)
sig = np.hstack([sig, np.zeros_like(sig)])

xout = None
y = np.array([])
for u in sig:
    yout, xout = ring_resonator.state_space.step(u, x0=xout)
    y = np.append(y, yout[-1])

plt.plot(np.abs(y)**2)
plt.show()

ring_resonator.compute_steady_state()

# sig = np.exp(1j*2*np.pi*t*0)
# sig1 = sig.reshape(-1, 1)
# sigs = np.hstack([1*sig1, 0])

# tout, yout, xout = lsim(ring_resonator.state_space_model, sigs, t)
# plt.plot(tout, np.abs(yout[:, 0])**2)
# plt.show()
pass

