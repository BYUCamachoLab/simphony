import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, fftfreq

# === Parameters ===
fs = 1e16               # Sampling frequency (Hz)
T = 10e-12                # Total duration (s)
N = int(T * fs)           # Number of samples
t = np.arange(N) / fs     # Time array
f0 = 193e12               # Carrier frequency (Hz)
linewidth = 40e12          # Lorentzian FWHM (Hz)

# === Phase noise model: Brownian motion ===
# Phase variance grows linearly with time: var(phi) = π * linewidth * t
# So, each time step gets Gaussian noise with std ~ sqrt(Δt * π * linewidth)
delta_phi_std = np.sqrt(2*np.pi * linewidth / fs)
dphi = np.random.randn(N) * delta_phi_std
phi = np.cumsum(dphi)  # Integrate to get phase

# === Generate signal: constant amplitude with noisy phase ===
E_t = np.exp(1j * (2 * np.pi * f0 * t + phi))

# === Compute spectrum ===
E_f = fftshift(fft(E_t))
freqs = fftshift(fftfreq(N, 1/fs))

# === Power spectral density (normalized) ===
psd = np.abs(E_f) ** 2
psd /= np.max(psd)  # Normalize for plotting

# === Plotting ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t * 1e3, np.real(E_t), label="Re")
plt.plot(t * 1e3, np.imag(E_t), label="Im", alpha=0.6)
plt.title("Time Domain Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t * 1e3, np.unwrap(phi))
plt.title("Cumulative Phase Noise")
plt.xlabel("Time (ms)")
plt.ylabel("Phase (rad)")

plt.subplot(2, 1, 2)
plt.plot(freqs / 1e3, psd)
# plt.plot(freqs/1e3, np.angle(E_f) ** 2)
plt.title("Power Spectrum (Lorentzian Linewidth)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Normalized PSD")
plt.grid(True)

plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# # --- Parameters ---
# sampling_freq = 1e16           # Hz
# dt = 1 / sampling_freq          # s
# duration = 10e-12                # s
# num_samples = int(duration / dt)
# t = np.arange(num_samples) * dt

# carrier_freq = 193.1e12        
# linewidth = 0                # Hz (FWHM)
# rin_level = 1e-16             # 1/Hz

# # --- Phase noise (Wiener process for FM noise) ---
# # linewidth = (1/2π) * d(φ^2)/dt → dφ ~ sqrt(2π * linewidth * dt)
# phase_noise_std = np.sqrt(2 * np.pi * linewidth * dt)
# phase_noise = np.cumsum(np.random.normal(0, phase_noise_std, size=num_samples))
# E_phase = np.exp(1j * (2 * np.pi * carrier_freq * t + phase_noise))

# # --- Amplitude noise (AM noise from RIN) ---
# # RIN is variance per Hz of power, so: std_amp ≈ sqrt(RIN × BW)
# bandwidth = sampling_freq / 2
# amp_std = np.sqrt(rin_level * bandwidth)
# amp_noise = np.random.normal(0, amp_std, size=num_samples)
# amplitude = 1.0 + amp_noise  # center around 1

# # --- Final complex field ---
# E_t = amplitude * E_phase

# # --- FFT ---
# freqs = np.fft.fftfreq(num_samples, dt)
# E_f = np.fft.fft(E_t)
# PSD = np.abs(E_f)**2
# PSD = PSD / np.max(PSD)  # Normalize for plotting

# # --- Plot ---
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(t * 1e6, np.real(E_t), label='Real part')
# plt.title("Time-Domain Signal")
# plt.xlabel("Time (µs)")
# plt.ylabel("Amplitude")

# plt.subplot(1, 2, 2)
# plt.plot(np.fft.fftshift(freqs) / 1e9, np.fft.fftshift(10 * np.log10(PSD)))
# plt.title("Frequency-Domain (FFT)")
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("PSD (dB, normalized)")

# plt.tight_layout()
# plt.show()

