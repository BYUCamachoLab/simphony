import numpy as np
import matplotlib.pyplot as plt
from math import log2

def generate_16qam_piecewise_linear_signal(T, dt, num_symbols, hold_time, ramp_time):
    """
    Generate a 16-QAM signal with piecewise linear transitions.
    
    Parameters:
      T            : Total duration (s)
      dt           : Time step (s)
      num_symbols  : Number of symbols
      hold_time    : Number of time steps to hold each symbol value
      ramp_time    : Number of time steps to linearly transition to the next symbol
      
    Returns:
      t              : Time vector (s)
      I_waveform     : In-phase component (length = expected_steps)
      Q_waveform     : Quadrature component (length = expected_steps)
      signal_complex : Complex baseband signal = I_waveform + j*Q_waveform
    """
    # Calculate expected total time steps
    expected_steps = num_symbols * (hold_time + ramp_time)
    
    # Use np.linspace to ensure exactly expected_steps samples.
    t = np.linspace(0, T, expected_steps, endpoint=False)
    
    # Optionally, you can check if expected_steps matches T/dt (approximately)
    total_steps_from_dt = int(round(T/dt))
    if total_steps_from_dt != expected_steps:
        print(f"Warning: T/dt (≈{total_steps_from_dt}) does not equal num_symbols*(hold_time+ramp_time) ({expected_steps}). Using expected_steps for t.")
    
    # 16-QAM: 4 bits per symbol.
    total_bits = num_symbols * 4
    data_bits = np.random.randint(0, 2, total_bits)
    
    # Mapping function for PAM4 (for both I and Q), Gray-like:
    # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    def bits_to_level(b0, b1):
        if (b0, b1) == (0, 0):
            return -3
        elif (b0, b1) == (0, 1):
            return -1
        elif (b0, b1) == (1, 1):
            return 1
        elif (b0, b1) == (1, 0):
            return 3

    symbols_I = []
    symbols_Q = []
    for i in range(0, total_bits, 4):
        bI0, bI1, bQ0, bQ1 = data_bits[i:i+4]
        symbols_I.append(bits_to_level(bI0, bI1))
        symbols_Q.append(bits_to_level(bQ0, bQ1))
    symbols_I = np.array(symbols_I)
    symbols_Q = np.array(symbols_Q)
    
    def piecewise_linear_hold(symbols, hold_time, ramp_time):
        """
        For each consecutive pair of symbols, hold the symbol value for 'hold_time' steps,
        then linearly ramp to the next symbol over 'ramp_time' steps.
        For the final symbol, hold it for the entire symbol period.
        """
        output = []
        for i in range(len(symbols) - 1):
            hold = np.full(hold_time, symbols[i])
            ramp = np.linspace(symbols[i], symbols[i+1], ramp_time, endpoint=False)
            block = np.concatenate([hold, ramp])
            output.append(block)
        # For the last symbol, hold it for a full symbol period.
        last_block = np.full(hold_time + ramp_time, symbols[-1])
        output.append(last_block)
        return np.concatenate(output)
    
    I_waveform = piecewise_linear_hold(symbols_I, hold_time, ramp_time)
    Q_waveform = piecewise_linear_hold(symbols_Q, hold_time, ramp_time)
    signal_complex = I_waveform + 1j * Q_waveform
    
    return t, I_waveform, Q_waveform, signal_complex

# -----------------------------
# Input Variables
# -----------------------------
T = 4e-11          # Total duration: 40 ps
dt = 1e-14         # Time step: 10 fs (should be close to 4000 total samples)
num_symbols = 40   # For example, 40 symbols so that 40*100 = 4000 time steps.
hold_time = 70     # Hold each symbol for 70 time steps.
ramp_time = 30     # Transition (ramp) for 30 time steps.

# Generate the signal
t, I_waveform, Q_waveform, signal_complex = generate_16qam_piecewise_linear_signal(T, dt, num_symbols, hold_time, ramp_time)

# -----------------------------
# Plot Time-Domain Waveforms for I(t) and Q(t)
# -----------------------------
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t*1e12, I_waveform, color='blue', label='I(t)')
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude")
plt.title("In-Phase Component I(t) with Linear Rise Time")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t*1e12, Q_waveform, color='red', label='Q(t)')
plt.xlabel("Time (ps)")
plt.ylabel("Amplitude")
plt.title("Quadrature Component Q(t) with Linear Rise Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2D Constellation Heat Map with Transition Paths
# -----------------------------
plt.figure(figsize=(6,6))
plt.hist2d(I_waveform, Q_waveform, bins=100, range=[[-4, 4], [-4, 4]], cmap='hot')
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation Heat Map with Linear Rise Time")
plt.colorbar(label="Density")
plt.axis('equal')
plt.show()


plt.figure(figsize=(6,6))
plt.plot(I_waveform, Q_waveform, color='blue', linewidth=1, alpha=0.7, label='Transition Path')
#plt.scatter(symbols_I, symbols_Q, color='red', s=50, zorder=5, label='Symbols')
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("16-QAM Constellation with Transition Paths")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()