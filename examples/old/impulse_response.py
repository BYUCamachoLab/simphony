from simphony.utils import discrete_time_impulse_response
import jax.numpy as jnp
import matplotlib.pyplot as plt

propagation_constants_0 = {
    # 1: 0.0,
    2: 1e-19, 
    3: 4e-35,
    4: 1e-47,
}
fs = 1e14

h0 = discrete_time_impulse_response(propagation_constants=propagation_constants_0, sampling_freq=fs)

propagation_constants_1 = {
    1: 1e-5,
    2: 1e-19, 
    3: 4e-35,
    4: 1e-47,
}

h1 = discrete_time_impulse_response(propagation_constants=propagation_constants_1, sampling_freq=fs)

plt.plot(jnp.abs(h0))
plt.plot(jnp.abs(h1))


plt.show()
