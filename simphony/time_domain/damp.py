import numpy as np
from scipy.signal import convolve, find_peaks, peak_widths
from simphony.time_domain.baseband_vector_fitting import BasebandModel
import jax.numpy as jnp
import warnings
from scipy.optimize import curve_fit

class QDamp():
    def __init__(self, delays, amplitudes, phases, initial_qstate=None):
        self.delays = delays
        self.amplitudes = amplitudes
        self.phases = phases
        self.initial_qstate =  initial_qstate

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (sigma**2))

def calculate_r_squared(y_data, y_fit):
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def find_gaussian_peaks(sig, std, window_size, r_squared_threshold=0.95):
    peaks, _ = find_peaks(sig)
    # peaks, _ = find_peaks(sig, height=0.0005)
    # peaks, _ = find_peaks(sig, height=0.0005)

    gaussian_peaks = []

    for peak in peaks:
        start = max(0, peak - window_size)
        end = min(len(sig), peak + window_size)
        x_data = np.arange(start, end)
        y_data = sig[start:end]
        
        gaussian_func = gaussian
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(gaussian_func, x_data, y_data, p0=[max(y_data), peak, window_size/2])
                amplitude, mu, sigma = popt
            
            y_fit = gaussian_func(x_data, *popt)
            r_squared = calculate_r_squared(y_data, y_fit)
            
            if r_squared >= r_squared_threshold:
                gaussian_peaks.append(peak)
            
        except RuntimeError:
            continue
    
    return gaussian_peaks

class Damp():
    def __init__(self, sig, std, pulse_index, dt):
        peaks = find_gaussian_peaks(np.abs(sig)**2, std, pulse_index, dt)
        t = np.linspace(0.0, dt * len(sig), len(sig))

        self.amps = np.array([])
        self.delays = np.array([])
        self.angles = np.array([])
        for peak in peaks:
            self.amps = np.append(self.amps, (np.abs(sig[peak])))
            self.angles = np.append(self.angles, np.angle(sig[peak]))
            delay = (peak - pulse_index) * dt
            self.delays = np.append(self.delays, delay)
        
        self.delay_indices = np.round(self.delays/dt).astype(int)
        self.num_pulses = self.delay_indices.shape[0]
        
    
    
    def compute_response(self, t, sig):
        K = t.shape[0]
        response = np.zeros(2 * K, dtype=complex)
        width = 90

        for y, i in zip(sig, range(len(sig))):
            if y > 0:
                pass
            # response_k = np.zeros(4 * t.shape[0], dtype=complex)
            for j in range(self.num_pulses):
                # print(y)
                index = i + self.delay_indices[j]
                # plt.plot(response)
                response[self.delay_indices[j] + i * width:self.delay_indices[j]+(i+1)*width] += np.abs(y)*np.exp(1j*np.angle(y))*self.amps[j]*np.exp(1j*self.angles[j])
                # response_k[self.delay_indices[j] + i * width:self.delay_indices[j]+(i+1)*width] += self.amps[j]*np.exp(1j*self.angles[j])

            # plt.plot(response_k)
        

        return response[0:K]

    def compute_impulse_responses(self, t, sig):
        K = t.shape[0]
        response = np.zeros(2 * K, dtype=complex)
        width = 1
        # responses = []
        responses = np.zeros((t.shape[0], sig.shape[0]), dtype=complex)
        for offset in range(t.shape[0]):
            delay_indices = self.delay_indices + offset
            delay_indices = delay_indices[delay_indices < len(t)]
            for p_ind in range(len(delay_indices)):
                d_ind = delay_indices[p_ind]
                y = sig[offset]
                # response[d_ind + width*offset:d_ind+width + width*offset] = np.abs(y) * np.exp(1j*np.angle(y))*self.amps[p_ind]*np.exp(1j*self.angles[p_ind])
                responses[offset, d_ind] = np.abs(y) * np.exp(1j*np.angle(y))*self.amps[p_ind]*np.exp(1j*self.angles[p_ind])
                # response[d_ind + width*offset:d_ind+width + width*offset] = self.amps[p_ind]*np.exp(1j*self.angles[p_ind])
        
        return responses


    


class DampModel():
    """Damp Model
    """

    def __init__(self, baseband_model: BasebandModel, T: float):
        self.system = baseband_model.generate_sys_discrete()
        self.dt = np.abs(self.system.dt)
        self.num_modes = self.system.B.shape[1]
        self.T = T
        self.K = round(self.T / self.dt)
        self.t = np.linspace(0.0, (self.K-1)*self.dt, self.K)
        self.damps = np.full((self.num_modes, self.num_modes), None) 
    
    def calculate_damps(self):
        std = 90*self.dt
        pulse = self.gaussian_pulse(self.t, 180*self.dt, std)
        pulse_index, _ = find_peaks(pulse)
        pulse_index = pulse_index[0]

        for i in range(self.num_modes):
            input_sig = np.zeros((self.K, self.num_modes), dtype=complex)
            input_sig[:, i] = pulse
            _, yout, _ = self.dlsim_complex(self.system, input_sig)
            for j in range(self.num_modes):
                self.damps[i, j] = Damp(yout[:, j], std, pulse_index, self.dt)
    

    def compute_response(self, input_sig):
        new_sig = np.zeros((self.K, self.num_modes), dtype=complex)
        new_sig[0:input_sig.shape[0]] = input_sig

        response = np.zeros((self.K, self.num_modes, self.num_modes), dtype=complex) 
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                response[:, i, j] = self.damps[i, j].compute_response(self.t, new_sig[:, i])

        return self.t, response
    
    def compute_impulse_responses(self, input_sig, a, b):
        new_sig = np.zeros((self.K, self.num_modes), dtype=complex)
        new_sig[0:input_sig.shape[0]] = input_sig

        return self.t, self.damps[a, b].compute_impulse_responses(self.t, input_sig[:, b])

    # def compute_response(self, t, input_sig):
    #     # t_orig = np.linspace(0.0, self.T, input_sig.shape[0])
    #     new_sig = np.zeros((self.K, self.num_modes), dtype=complex)
    #     for n in range(self.num_modes):
    #         new_sig[:, n] = np.interp(self.t, t, input_sig[:, n])


    #     response = np.full((self.num_modes, self.num_modes), None) 
    #     for i in range(self.num_modes):
    #         for j in range(self.num_modes):
    #             response[i, j] = self.damps[i, j].compute_response(self.t, new_sig[:, i])

    #     return response


    @staticmethod
    def gaussian_pulse(t, t0, std, a=1.0 ):
        return a * jnp.exp(-(t - t0)**2 / std**2) 

    @staticmethod
    def dlsim_complex(system, u, t=None, x0=None):
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