"""Ideal time-domain models."""
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from simphony.time_domain.time_system import TimeSystem
import simphony.libraries.ideal as fd
from simphony.utils import dict_to_matrix, mul_polar
from queue import Queue

class TimeCoupler(TimeSystem):
    def __init__(
        self, 
        coupling: float = 0.5, 
        loss: float = 0.0, 
        phi: float = jnp.pi / 2,
    ) -> None:
        super().__init__()
        self.s_params = dict_to_matrix(fd.coupler(coupling=coupling, loss=loss, phi=phi))
        self.num_ports = 4
        pass

    def response(self, input: ArrayLike) -> ArrayLike:
        output = jnp.zeros((self.num_ports, input.shape[1]), dtype=complex)
        for i in range(self.num_ports):
            output = output.at[i, :].set(input[0, :] * self.s_params[0, i, 0]
                                         + input[1, :] * self.s_params[0, i, 1]
                                         + input[2, :] * self.s_params[0, i, 2]
                                         + input[3, :] * self.s_params[0, i, 3]
                                         )
        return output

class TimeWaveguide(TimeSystem):
    def __init__(
        self,
        # wl: ArrayLike | float = 1.55,
        dt: float,
        wl0: float = 1.55,
        neff: float = 2.34,
        ng: float = 3.4,
        length: float = 10.0,
        loss: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_ports = 2
        c = 299792458
        group_velocity = c / ng
        omega = c / (1.55e-6)

        delay = (length * 1e-6) / (group_velocity)
        num_delay_indices = round(delay / dt)

        phase_shift = jnp.mod(omega*delay, 2*jnp.pi)
        loss_mag = loss / (10 * jnp.log10(jnp.exp(1)))
        alpha = loss_mag * 1e-4
        amplitude = jnp.asarray(jnp.exp(-alpha * length / 2), dtype=complex)

        self.transmission = amplitude * jnp.exp(1j * phase_shift)
        self.forward_wave = Queue()
        self.backward_wave = Queue()

        for i in range(num_delay_indices):
            self.forward_wave.put(0+0j)
            self.backward_wave.put(0+0j)

    def response(self, input: ArrayLike) -> ArrayLike:
        output = jnp.zeros((self.num_ports, input.shape[1]), dtype=complex)

        for i in range(input.shape[1]):
            a = complex(input[0, i])
            b = complex(input[1, i])
            output = output.at[0, i].set(self.transmission * self.forward_wave.get())
            output = output.at[1, i].set(self.transmission * self.backward_wave.get())

            self.forward_wave.put(a)
            self.backward_wave.put(b)
        
        return output
