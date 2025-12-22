import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000
T = 1.0  # seconds
sigma = 20  # Hz -- this controls spectral width
t = np.linspace(0, T, N)

# Sample a single Gaussian variable
Z = np.random.randn()

# Generate phi(t) = 2 * pi * sigma * t * Z
phi_t = 2 * np.pi * sigma * t * Z

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, phi_t, label=r'$\phi(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Phase')
plt.title('Phase noise with quadratic variance: Var[$\phi(t)$] ∝ $t^2$')
plt.grid(True)
plt.tight_layout()
plt.show()
