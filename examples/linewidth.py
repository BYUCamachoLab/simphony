import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, fftfreq

# === Parameters ===
fs = 1e6                  # Sampling frequency (Hz)
T = 1e-3                  # Total duration (s)
N = int(T * fs)           # Number of samples
t = np.arange(N) / fs     # Time array
f0 = 1e5                  # Carrier frequency (Hz)
linewidth = 1e3           # Lorentzian FWHM (Hz)

# === Phase noise model: Brownian motion ===
# Phase variance grows linearly with time: var(phi) = π * linewidth * t
# So, each time step gets Gaussian noise with std ~ sqrt(Δt * π * linewidth)
delta_phi_std = np.sqrt(np.pi * linewidth / fs)
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
plt.title("Power Spectrum (Lorentzian Linewidth)")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Normalized PSD")
plt.grid(True)

plt.tight_layout()
plt.show()
