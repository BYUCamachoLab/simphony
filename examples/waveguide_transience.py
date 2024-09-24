import numpy as np
import matplotlib.pyplot as plt
import sax

from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix

from simphony.baseband_vector_fitting import Baseband_Model_SingleIO, BVF_Options
from scipy.signal import  StateSpace, dlsim, lsim

from simphony.quantum import QuantumState, CoherentState, SqueezedState, compose_qstate

circuit, info = sax.circuit(
    netlist= {
        "instances": {
            "wg1": "waveguide",
        },
        "connections": {
        },
        "ports": {
            "o0": "wg1,o0",
            "o1": "wg1,o1",
        }
    },
    models={
        "waveguide": ideal.waveguide,
    }
)

num_measurements = 200
wvl = np.linspace(1.5, 1.6, num_measurements)
center_wvl = 1.55
ng = 3.4
neff = 2.34

c = 299792458
l = 150.0e-6
s = circuit(wl=wvl, 
            wg1={"length": l*1e6, "loss": 100, "neff": neff, "ng": ng, "wl0": center_wvl})
S = dict_to_matrix(s)
S = np.asarray(S)

model_order = 50
beta = 10.0
# beta = 3.0
options = BVF_Options(max_iterations=5, beta=beta, gamma=0.95) # 3 iterations
model = Baseband_Model_SingleIO(wvl, center_wvl, S[:, 0, 1], model_order, options)

model.fit_model()
model.compute_state_space_model()

sys = StateSpace(model.A, model.B, model.C, model.D, dt = 1* 1/model.sampling_freq)


N = int(10000)
T = 1.0e-12
t = np.linspace(0, T, N)

sig = np.exp(1j*2*np.pi*t*0)

delta = 0.01
A = 1
f = 1/(T)
width = 1.6
# y1 = A*sin(2*pi*t*f)
smoothed_square = (A/np.pi)*np.arctan(np.sin(((1 / width)*2*np.pi)*t*f - np.pi/2 + 1.2)/delta) + 0.5
# plt.plot(t, smoothed_square)
# plt.show()
# y3 = (A/atan(1/delta))*atan(sin(2*pi*t*f)/delta)

# sig[:N//2] *= 1
# sig[N//2:] *= 1
sig1 = 0 * sig
sig1 = sig1.reshape(-1, 1)
u1 = np.hstack([np.real(sig1), np.imag(sig1)])

sig2 = 1 * sig * smoothed_square
sig2 = sig2.reshape(-1, 1)
u2 = np.hstack([np.real(sig2), np.imag(sig2)])

sig3 = 0 * sig
sig3 = sig3.reshape(-1, 1)
u3 = np.hstack([np.real(sig3), np.imag(sig3)])

sig4 = 0 * sig
sig4 = sig4.reshape(-1, 1)
u4 = np.hstack([np.real(sig4), np.imag(sig4)])

sig5 = 0 * sig
sig5 = sig5.reshape(-1, 1)
u5 = np.hstack([np.real(sig5), np.imag(sig5)])

tout1, yout1, xout1 = dlsim(sys, u1, t)
tout2, yout2, xout2 = dlsim(sys, u2, t, xout1[-1, :])
tout3, yout3, xout3 = dlsim(sys, u3, t, xout2[-1, :])
tout4, yout4, xout4 = dlsim(sys, u4, t, xout3[-1, :])
tout5, yout5, xout5 = dlsim(sys, u5, t, xout4[-1, :])

# plt.plot(np.abs(yout1[:, 0] + 1j*yout1[:, 1])**2)
# plt.show()

# plt.plot(np.abs(yout2[:, 0] + 1j*yout2[:, 1])**2)
# plt.show()

# plt.plot(np.abs(yout3[:, 0] + 1j*yout3[:, 1])**2)
# plt.show()

# plt.plot(np.abs(yout4[:, 0] + 1j*yout4[:, 1])**2)
# plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
# ax[1].plot(x, y2)

tout = np.linspace(0, 5*T, len(tout1) + len(tout2) + len(tout3) + len(tout4) + len(tout5))
yout = np.vstack([yout1, yout2, yout3, yout4, yout5])
u = np.vstack([u1, u2, u3, u4, u5])
t = np.linspace(0, 5*T, len(u))
ax[0].plot(t[:int(5*N/2.2)], np.abs(u[:int(5*N/2.2), 0] + 1j*u[:int(5*N/2.2), 1])**2, label="Input")
ax[0].plot(tout[int(len(yout)/2.2):], np.abs(yout[int(len(yout)/2.2):, 0] + 1j*yout[int(len(yout)/2.2):, 1])**2, label="Response")
# plt.axvline(x=1*T, color='b', linestyle='--', linewidth=1.2, alpha=0.7)
# plt.axvline(x=2*T, color='b', linestyle='--', linewidth=1.2, alpha=0.7)
ax[0].axvline(x=ng*l/c + 1*T, color='r', linestyle='--', linewidth=1.2, alpha=0.7)
ax[0].axvline(x=ng*l/c + 2*T, color='r', linestyle='--', linewidth=1.2, alpha=0.7)
ax[0].axhline(y=0.70795, color='black', linewidth=6.0, alpha= 0.2)
ax[0].set_xlabel("Time")
ax[0].set_ylabel("E-field Amplitude")
ax[0].legend(loc="center left")
ax[0].set_title(f" Waveguide Transience")

# model.plot_model()
freq = np.linspace(-model.sampling_freq / 10, model.sampling_freq / 10, int(model.options.beta * num_measurements))
ax[1].plot(freq + model.center_freq, np.abs(model.compute_response(freq=freq))**2, label=f"Model")
ax[1].plot(model.freqs + model.center_freq, np.abs(model.S)**2, "r--", label="Samples")
ax[1].axhline(y=0.70795, color='black', linewidth=6.0, alpha= 0.12)
ax[1].set_xlabel("Frequency")
ax[1].set_ylabel("Transmission")
ax[1].legend(loc="center left")
ax[1].set_title(f" Waveguide S-params:\nlength={l*1e6}um")

plt.show()


in_state = CoherentState("o0", 1.0 + 0.0j)
in_state = SqueezedState("o0", r=1.0, phi=0, alpha=5.0 + 5.0j)
vac = CoherentState("o1", 0.0 + 0.0j)

qstate = compose_qstate(in_state, vac)
qstate = compose_qstate(in_state)
qstate._add_vacuums(1)
qstate.to_xxpp()

N = int(10000)
input_states = [qstate for i in range(N)]

T = 200.0e-12

t = np.linspace(0, T, N)
sig = np.exp(1j*2*np.pi*t*0).reshape(-1, 1)
sig = np.hstack([sig, np.zeros_like(sig)])

def realify(M):
    top_row = np.hstack([M.real, -M.imag])
    bot_row = np.hstack([M.imag,  M.real])
    M_real = np.vstack([top_row, bot_row])
    return M_real

A = realify(sys.A)
B = realify(sys.B)
C = realify(sys.C)
D = realify(sys.D)

in_state = input_states[0]

x = np.zeros_like(A)
x = A@x@A.T + B@in_state.cov@B.T
cov = []
for i in range(0, 10000):
    x = A@x@A.T + B@in_state.cov@B.T
    y = C@x@C.T + D@in_state.cov@D.T
    cov.append(y)

cov = np.stack(cov)

plt.plot(cov[:, 0,0])
plt.plot(cov[:, 2,2])
plt.show()

pass

