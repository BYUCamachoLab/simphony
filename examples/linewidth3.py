import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.fft import fft, fftfreq, fftshift

# Simulation parameters
fs = 10e7            # Sampling frequency (Hz)
T = 100e-6            # Total time (s)
N = int(T * fs)      # Number of points
t = np.linspace(0, T, N, endpoint=False)
f0 = 0            # Laser central frequency (Hz)
linewidth = 1e6

# === Lorentzian PSD ===
# Phase = Wiener process (integrated white noise)
np.random.seed(0)
dphi_lorentz = np.random.normal(scale=np.sqrt(2 * np.pi * linewidth / fs), size=N)
phi_lorentz = np.cumsum(dphi_lorentz)  # Integrate to get phase
E_lorentz = np.exp(1j * (2 * np.pi * f0 * t + phi_lorentz))

# === Gaussian PSD ===
# Phase = bandlimited white noise (convolved with Gaussian)
noise = np.random.normal(scale=np.sqrt(2 * np.pi * linewidth / fs), size=N)
# noise = phi_lorentz
window_width = int(N // 100)  # Control PSD width
gaussian_window = gaussian(N, std=window_width/2)
gaussian_window /= np.sum(gaussian_window)
phi_gauss = np.convolve(noise, gaussian_window, mode='same')
phi_gauss *= 200 # scale for similar phase noise strength
E_gauss = np.exp(1j * (2 * np.pi * f0 * t + phi_gauss))

# === FFT Analysis ===
def compute_psd(E):
    E_f = fftshift(fft(E))
    freqs = fftshift(fftfreq(N, 1/fs))
    psd = np.abs(E_f)**2
    psd /= np.max(psd)  # Normalize
    return freqs, psd

freqs, psd_lorentz = compute_psd(E_lorentz)
_, psd_gauss = compute_psd(E_gauss)

# === Plot Time Domain Phase ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t * 1e6, phi_lorentz, label='Lorentzian Phase')
plt.plot(t * 1e6, phi_gauss, label='Gaussian Phase')
plt.xlabel('Time (µs)')
plt.ylabel('Phase (rad)')
plt.title('Phase Noise')
plt.legend()

# === Plot Frequency Domain ===
plt.subplot(1, 2, 2)
# plt.plot(freqs / 1e9, 10 * np.log10(psd_lorentz + 1e-12), label='Lorentzian PSD')
# plt.plot(freqs / 1e9, 10 * np.log10(psd_gauss + 1e-12), label='Gaussian PSD')
plt.plot(freqs / 1e9, 10 * psd_lorentz + 1e-12, label='Lorentzian PSD')
plt.plot(freqs / 1e9, 10 * psd_gauss + 1e-12, label='Gaussian PSD')
plt.xlabel('Frequency (GHz)')
plt.ylabel('PSD (dB)')
plt.title('Power Spectral Density')
plt.legend()
plt.tight_layout()
plt.show()
