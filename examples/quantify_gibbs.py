import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_terms = 10  # Number of Fourier series terms
x = np.linspace(-np.pi, np.pi, 1000)
square_wave = np.sign(np.sin(x))  # Ideal square wave

# Fourier series approximation
approx_square_wave = np.zeros_like(x)
for n in range(1, n_terms * 2, 2):
    approx_square_wave += (4 / (np.pi * n)) * np.sin(n * x)

# Identify the discontinuity region (just before and after the jump)
discontinuity_index = np.argmax(x >= 0)  # Change from negative to positive in square wave

# Measure the overshoot
overshoot_value = np.max(approx_square_wave[discontinuity_index-10:discontinuity_index+10])
ideal_value = 1  # The ideal value after the discontinuity for the square wave

# Calculate the Gibbs overshoot
gibbs_overshoot = overshoot_value - ideal_value
percentage_overshoot = (gibbs_overshoot / (ideal_value - (-1))) * 100

print(f"Gibbs overshoot value: {gibbs_overshoot}")
print(f"Percentage overshoot: {percentage_overshoot}%")

# Plot the signal to visualize
plt.plot(x, approx_square_wave, label='Fourier Approximation')
plt.axhline(y=1, color='r', linestyle='--', label='Ideal Square Wave Value')
plt.axhline(y=overshoot_value, color='g', linestyle='--', label='Overshoot Value')
plt.legend()
plt.show()
