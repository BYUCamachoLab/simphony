import jax.numpy as jnp
import matplotlib.pyplot as plt
import sax

from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries import ideal
from simphony.utils import dict_to_matrix
from simphony.time_domain.baseband_vector_fitting import BasebandModel
from simphony.time_domain.damp import DampModel
from scipy.signal import  StateSpace, dlsim
from simphony.utils import gaussian_pulse

wl = jnp.linspace(1.5, 1.6, 1000)
length = 100
loss = 20
neff = 2.34
ng = 3.4
wl0 = 1.55

wg = ideal.waveguide(wl=wl, length=length, loss=loss, neff=neff, ng=ng, wl0= wl0)
s = dict_to_matrix(wg)

baseband_model  = BasebandModel(wl, wl0, s, 100)
response = baseband_model.compute_response(1, 0)

plt.plot(wl, jnp.abs(response)**2)
plt.plot(wl, jnp.abs(s[:, 1, 0])**2)
plt.show()

sys = baseband_model.generate_sys_discrete()

N = int(10000)
T = 3.0e-12
t = jnp.linspace(0, T, N)
sig = jnp.exp(1j*2*jnp.pi*t*0)

# y1 = A*sin(2*pi*t*f)
gaussian = gaussian_pulse(t, T/4, 0.1e-12)

sig1 = 0 * gaussian
sig2 = 1 * sig * gaussian

tout, yout, xout = dlsim(sys, jnp.array([sig1, sig2]).T, t)

plt.plot(t, jnp.abs(sig2)**2)
plt.plot(tout, jnp.abs(yout)**2)
plt.show()

loss_mag = loss / (10 * jnp.log10(jnp.exp(1)))
alpha = loss_mag * 1e-4
phase = 2 * jnp.pi * (neff - (wl - wl0) * (ng - neff) / wl0) * length / wl
amplitude = jnp.asarray(jnp.exp(-alpha * length / 2), dtype=complex)
H = amplitude * jnp.exp(1j * phase)
plt.plot(H)
plt.show()

# baseband_model.plot_model()

pass