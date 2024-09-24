from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import sax
import pandas as pd

from simphony.quantum import QuantumTimeElement
from simphony.libraries import ideal
from simphony.utils import smooth_rectangular_pulse, dict_to_matrix
from simphony.baseband_vector_fitting import BasebandModelSingleIO, CVF_Model

from simphony.baseband_vector_fitting import BasebandModel
from scipy.signal import correlate


def calculate_transience(system):
    dt = system.dt
    n = system.B.shape[1]
    K = 1000 # number of time steps
    t = np.linspace(0, K*dt, K)

    pulse = smooth_rectangular_pulse(t, t_start=1e-12, t_end=2e-12)

    input_signal = np.zeros((K, n), dtype=complex)
    input_signal[:, 0] = pulse
    input_signal[:, 1] = 0.0 + 0.0*1j


    t, y, _ = mydlsim(system, input_signal)

    plt.plot(t, np.abs(input_signal[:, 0])**2)
    plt.plot(t, np.abs(y[:, 1])**2)
    plt.show()

    # Compute the cross-correlation
    correlation = correlate(np.abs(y[:, 1])**2, np.abs(input_signal[:, 0])**2, mode='full')

    # Find the index of the maximum cross-correlation
    lag_index = np.argmax(correlation) - (len(input_signal) - 1)
    time_delay = lag_index * dt

    return time_delay

def mydlsim(system, u, t=None, x0=None):
    out_samples = len(u)
    stoptime = (out_samples - 1) * system.dt

    xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
    yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
    tout = np.linspace(0.0, stoptime, num=out_samples)

    xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)

    u_dt = u

    # Simulate the system
    for i in range(0, out_samples - 1):
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]))
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))

    # Last point
    yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                              np.dot(system.D, u_dt[out_samples-1, :]))

    return tout, yout, xout

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


ng = 3.4
neff = 2.34
wl = np.linspace(1.5, 1.6, 100)
wl0 = 1.55

ckt = circuit(wl=wl, wg={"length": 500, "loss": 50})
s_params = np.copy(np.asarray(dict_to_matrix(ckt)))
model = BasebandModel(wl, wl0, s_params, 50)

sys = model.generate_sys_discrete()

delay = calculate_transience(sys)
print(f"delay: {delay}")
pass
