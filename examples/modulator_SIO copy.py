import numpy as np
import matplotlib.pyplot as plt
import sax
import random

from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal, siepic
from simphony.utils import dict_to_matrix

from simphony.baseband_vector_fitting import Baseband_Model_SingleIO, BVF_Options
from scipy.signal import  StateSpace, dlsim, lsim

def eye_diagram(t, signal, eye_period):
    # t_period = np.linspace(0, eye_period, len(t))
    num_periods = int(t[-1] // eye_period)
    num_indices = int(len(t) // num_periods)
    indices = range(0, num_indices)

    tout = t[indices]
    yout = signal[indices]
    for i in range(1, num_periods):
        indices = range(num_indices * i, num_indices * (i+1))
        # tout = np.vstack((tout, tout))
        yout = np.vstack((yout, signal[indices]))
    
    tout = np.tile(tout, (num_periods, 1)).T
    yout = yout.T

    return tout, yout


def step(system, u, t=None, x0=None):
    # Condition needed to ensure output remains compatible
    is_ss_input = isinstance(system, StateSpace)

    # Check initial condition
    if x0 is None:
        xout = np.zeros((system.A.shape[1],))
    else:
        xout = np.asarray(x0)

    # Simulate the system
    xout = (np.dot(system.A, xout) +
                        np.dot(system.B, u))
    yout = (np.dot(system.C, xout) +
                      np.dot(system.D, u))

    return yout, xout

def modulated_response(sig_real_valued, T, phi, wvl0, ring_length, ng, period, rise_time, fall_time):
    xout_coupler10 = None
    xout_coupler12 = None
    xout_coupler30 = None
    xout_coupler32 = None

    yout_wg = np.zeros(1)
    yout_wg = np.hstack([np.real(yout_wg), np.imag(yout_wg)])
    xout_wg = np.zeros(2*model_order)
    yout_total = np.array([])
    counter = 0
    neff=2.34
    wg = dict_to_matrix(ideal.waveguide(wl=wvl, length=ring_length, loss=100, neff=neff, ng=ng, wl0=center_wvl))
    model_wg = Baseband_Model_SingleIO(wvl, center_wvl, wg[:, 0, 1], model_order, options)
    model_wg.fit_model()
    model_wg.compute_state_space_model()

    sys_wg = StateSpace(model_wg.A, model_wg.B, model_wg.C, model_wg.D, dt=1/model_wg.sampling_freq)

    phi_vectorized = np.vectorize(phi)
    phase_shifts = phi_vectorized(np.linspace(0, T, len(sig_real_valued)), period=period, rise_time=rise_time, fall_time=fall_time)

    noise = np.random.normal(0, 0.002 * np.pi, len(phase_shifts))
    phase_shifts += noise

    for input_signal, phase_shift in zip(sig_real_valued, phase_shifts):
        progress = counter / len(sig_real_valued)
        counter += 1
        print(progress)

        yout_coupler10, xout = step(sys_coupler10, input_signal,  x0=xout_coupler10)
        xout_coupler10 = xout

        yout_coupler12, xout = step(sys_coupler12, yout_wg, x0=xout_coupler12)
        xout_coupler12 = xout

        yout_coupler30, xout = step(sys_coupler30, input_signal, x0=xout_coupler30)
        xout_coupler30 = xout

        yout_coupler32, xout = step(sys_coupler32, yout_wg, x0=xout_coupler32)
        xout_coupler32 = xout

        yout_wg, xout = step(sys_wg, yout_coupler32 + yout_coupler30, x0=xout_wg)
        xout_wg = xout
        yout_wg = np.abs(yout_wg[0] + 1j*yout_wg[1])*np.exp(1j*(np.angle(yout_wg[0] + 1j*yout_wg[1]) + phase_shift))
        yout_wg = [np.real(yout_wg), np.imag(yout_wg)]

        yout_total = np.append(yout_total, yout_coupler10[0] + 1j*yout_coupler10[1] + (yout_coupler12[0] + 1j*yout_coupler12[1]))
    
    return yout_total, phase_shifts

def phi(t, period=50e-12, rise_time=80.0e-13, fall_time=None):
    if rise_time is None:
        rise_time = 0.0
    if fall_time is None:
        fall_time = rise_time

    A = np.pi
    
    if t < 0*period/2 + rise_time:
        t0 = t - 0*period/2
        return A * t0 / rise_time
    elif t < 1*period/2:
        return A 
    elif t < 1*period/2+fall_time:
        t0 = t - 1*period/2
        return A * (1 - t0 / fall_time)
    elif t < 2*period/2:
        return 0 
    elif t < 2*period/2+rise_time:
        t0 = t - 2*period/2
        return A * t0 / rise_time
    elif t < 3*period/2:
        return A
    elif t < 3*period/2+fall_time:
        t0 = t - 3*period/2
        return A * (1 - t0 / fall_time)
    elif t < 4*period/2:
        return 0
    elif t < 4*period/2+rise_time:
        t0 = t - 4*period/2
        return A * t0 / rise_time
    elif t < 5*period/2:
        return A
    elif t < 5*period/2+fall_time:
        t0 = t - 5*period/2
        return A * (1 - t0 / fall_time)
    elif t < 6*period/2:
        return 0
    elif t < 6*period/2+rise_time:
            t0 = t - 6*period/2
            return A * t0 / rise_time
    elif t < 7*period/2:
        return A
    elif t < 7*period/2+fall_time:
        t0 = t - 7*period/2
        return A * (1 - t0 / fall_time)
    elif t < 8*period/2:
        return 0
    
    # if t < 1*period + rise_time:
    #     t0 = t - 1*period
    #     return A * t0 / (rise_time)
    # if t < 1*period:
    #     return A 



num_measurements = 190
wvl = np.linspace(1.5, 1.6, num_measurements)

center_wvl = 1.5355
neff = 2.34
ng = 3.4
c = 299792458
coupler = np.tile(dict_to_matrix(siepic.bidirectional_coupler()), (num_measurements, 1, 1))
coupler = np.asarray(coupler)
# coupler = dict_to_matrix(siepic.bidirectional_coupler(wl=wvl))

model_order = 55
options = BVF_Options(max_iterations=5, beta=7.0, gamma=0.95) # 3 iterations

model_coupler10 = Baseband_Model_SingleIO(wvl, center_wvl, coupler[:, 3, 1], model_order, options)
model_coupler12 = Baseband_Model_SingleIO(wvl, center_wvl, coupler[:, 3, 0], model_order, options)
model_coupler30 = Baseband_Model_SingleIO(wvl, center_wvl, coupler[:, 2, 1], model_order, options)
model_coupler32 = Baseband_Model_SingleIO(wvl, center_wvl, coupler[:, 2, 0], model_order, options)

model_coupler10.fit_model()
model_coupler12.fit_model()
model_coupler30.fit_model()
model_coupler32.fit_model()

model_coupler10.compute_state_space_model()
model_coupler12.compute_state_space_model()
model_coupler30.compute_state_space_model()
model_coupler32.compute_state_space_model()

sys_coupler10 = StateSpace(model_coupler10.A, model_coupler10.B, model_coupler10.C, model_coupler10.D, dt = 1/model_coupler10.sampling_freq)
sys_coupler12 = StateSpace(model_coupler12.A, model_coupler12.B, model_coupler12.C, model_coupler12.D, dt = 1/model_coupler12.sampling_freq)
sys_coupler30 = StateSpace(model_coupler30.A, model_coupler30.B, model_coupler30.C, model_coupler30.D, dt = 1/model_coupler30.sampling_freq)
sys_coupler32 = StateSpace(model_coupler32.A, model_coupler32.B, model_coupler32.C, model_coupler32.D, dt = 1/model_coupler32.sampling_freq)

N = int(1000)
# T = 300.0e-12
T = 1200.0e-12


time_step = 1/model_coupler10.sampling_freq
t = np.arange(0, T, time_step)

sig = np.exp(1j*2*np.pi*t*0)
sig1 = sig.reshape(-1, 1)
sig_real_valued = np.hstack([np.real(sig1), np.imag(sig1)])

ring_length = 75.0

modulator_period = 24e-12
modulator_frequency = 1 / modulator_period
rise_time = 0.5e-12
fall_time = 0.5e-12
y, phase_shifts = modulated_response(sig_real_valued, T, phi, center_wvl, ring_length, ng, modulator_period, rise_time, fall_time)

t = np.linspace(0, T, len(y))
plt.plot(t, np.abs(y)**2, label="Amplitude")
plt.plot(t, phase_shifts, label="phi(t)", linestyle='--')
plt.title(f"Modulated Time Response: {(modulator_frequency/1e9):.1f}GHz")
plt.xlabel("time")
plt.ylabel("Transmission")
plt.legend()
plt.show()


# plt.scatter(np.real(y), np.imag(y), color='blue', marker='.')
# plt.show()


t = np.linspace(0, T, len(y))
tout, yout = eye_diagram(t, y, modulator_period)
plt.plot(tout, np.abs(yout)**2)
plt.show()

# steady_state =[]
# phase_shifts = np.linspace(-2*np.pi, 2*np.pi, 150)
# for phase_shift in phase_shifts:
#     yout_shifted = modulated_response(sig_real_valued, T, phi, center_wvl, ring_length, ng)

#     tout = np.linspace(0, T, len(yout_shifted))
#     stepping_time = ring_length * 1e-6 / (c / ng)
#     for i in range (20):
#         plt.axvline(i * stepping_time, color='r', linestyle='--', linewidth=1, alpha=0.75)
    

#     steady_state.append(yout_shifted[-1])
#     plt.plot(tout, np.abs(yout_shifted)**2)
#     plt.axhline(np.abs(steady_state[-1])**2, color='orange', linestyle= "--", label=f"Steady-State={np.abs(steady_state[-1])**2:.2f}, alpha=0.1")
#     plt.title(f"Time Response: {center_wvl:.2f}um")
#     plt.xlabel("time")
#     plt.ylabel("Transmission")
#     plt.legend()
#     plt.show()
# plt.plot(phase_shifts, np.abs(steady_state)**2)
