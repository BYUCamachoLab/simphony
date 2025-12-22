import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100e9               # Sampling rate (100 GHz)
T = 10e-6                 # Total time duration (10 us)
N = int(T * fs)           # Number of samples
t = np.arange(N) / fs     # Time array

f0 = 0                    # Carrier frequency offset (can be nonzero)
linewidth_fwhm = 10e9     # Gaussian linewidth (FWHM) = 1 MHz

# Convert FWHM to standard deviation
sigma_f = linewidth_fwhm / (2 * np.sqrt(2 * np.log(2)))

# Frequency vector
freqs = np.fft.fftfreq(N, d=1/fs)

# Gaussian PSD centered at f0
psd = np.exp(- (freqs - f0)**2 / (2 * sigma_f**2))

# Generate random complex spectrum with Gaussian envelope
np.random.seed(0)
phases = np.exp(1j * 2 * np.pi * np.random.rand(N))
spectrum = np.sqrt(psd) * phases

# Ensure Hermitian symmetry for real signal (optional: for real-valued output)
# spectrum[N//2+1:] = np.conj(spectrum[1:N//2][::-1])

# Inverse FFT to get time domain signal
signal = np.fft.ifft(spectrum)

# Normalize
signal /= np.max(np.abs(signal))

# Plot time domain signal
plt.figure(figsize=(12, 4))
plt.plot(t[:1000]*1e9, np.real(signal[:1000]), label='Real part')
plt.plot(t[:1000]*1e9, np.imag(signal[:1000]), label='Imag part')
plt.title("Laser Electric Field (Gaussian Linewidth, Time Domain)")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot spectrum
plt.figure(figsize=(12, 4))
# plt.plot(np.fft.fftshift(freqs)/1e6, 10*np.log10(np.fft.fftshift(np.abs(spectrum)**2)))
plt.plot(np.fft.fftshift(freqs)/1e6, np.fft.fftshift(np.abs(spectrum)**2))
plt.title("Power Spectral Density (Gaussian)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("PSD (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
