import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp






data = np.load("simphony/16QAMGen.npz", allow_pickle=True)    
t = data["t"]

def upsample_trajectory(I, Q, factor=20):
    I_list, Q_list = [], []
    n = len(I)
    for i in range(n - 1):
        i0, i1 = I[i], I[i+1]
        q0, q1 = Q[i], Q[i+1]
        for alpha in np.linspace(0, 1, factor, endpoint=False):
            I_list.append(i0 + alpha*(i1 - i0))
            Q_list.append(q0 + alpha*(q1 - q0))
    # Add the last point
    I_list.append(I[-1])
    Q_list.append(Q[-1])
    return np.array(I_list), np.array(Q_list)


I_out = data["I"]
Q_out = data["Q"]
phase_array1 = data["phase_array1"]
phase_array2 = data["phase_array2"]
phase_array3 = data["phase_array3"]
phase_array4 = data["phase_array4"]

complex_signal = I_out + 1j*Q_out

t_ps = t * 1e12

I_out_up, Q_out_up = I_out, Q_out


plt.figure(figsize=(8,6))
bins = 500
plt.hist2d( Q_out_up,I_out_up, bins=bins, cmap='jet',norm=matplotlib.colors.LogNorm() )
plt.colorbar(label="Counts per bin")
plt.title("Output Constellation Diagram (Port 2))")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(12, 3 * 2), squeeze=False)
fig.suptitle('Decomposed Complex Output Signal (I, Q) at Port 2', fontsize=18)

# I‑trace
axs[0, 0].plot(t_ps, I_out, color='blue', lw=2, label='I_out')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].set_title("I vs. Time")
axs[0, 0].set_xlabel("Time (ps)")
axs[0, 0].set_ylabel("Amplitude")
axs[0,0].plot(t_ps+0.4, phase_array1/jnp.pi/2, color='green', lw=2, label='Modulating Signal(PM 1)',linestyle='--')
axs[0,0].plot(t_ps+0.4, phase_array2/jnp.pi/2, color='orange', lw=2, label='Modulating Signal(PM 2)',linestyle='--') 
axs[0, 0].legend(loc='upper right') 
# Q‑trace
axs[1, 0].plot(t_ps, Q_out, color='red', lw=2, label='Q_out')
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].set_title("Q vs. Time")
axs[1, 0].set_xlabel("Time (ps)")
axs[1, 0].set_ylabel("Amplitude")
axs[1,0].plot(t_ps+0.4, phase_array3/jnp.pi/2, color='green', lw=2, label='Modulating Signal(PM 3)',linestyle='--')
axs[1,0].plot(t_ps+0.4, phase_array4/jnp.pi/2, color='orange', lw=2, label='Modulating Signal(PM 4)',linestyle='--')  
axs[1, 0].legend(loc='upper right')

for ax in axs.flat:
    ax.set_xlim(25, 75)

fig.subplots_adjust(top=0.88)

plt.tight_layout()
plt.show()
