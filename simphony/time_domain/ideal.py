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
        self.ports = ['o0', 'o1', 'o2', 'o3']
        pass

    def response(self, inputs: dict) -> dict:
        N = inputs['o0'].shape
        response = {
            'o0': jnp.zeros((N), dtype=complex),
            'o1': jnp.zeros((N), dtype=complex),
            'o2': jnp.zeros((N), dtype=complex),
            'o3': jnp.zeros((N), dtype=complex),
        }
        
        for i in range(self.num_ports):
            response[f'o{i}'] = ( inputs['o0'] * self.s_params[0, i, 0]
                               + inputs['o1'] * self.s_params[0, i, 1]
                               + inputs['o2'] * self.s_params[0, i, 2]
                               + inputs['o3'] * self.s_params[0, i, 3])
        return response

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
        self.ports = ['o0', 'o1']
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

    def response(self, inputs: dict) -> dict:
        N = inputs['o0'].shape[0]
        o0_response = jnp.zeros((N), dtype=complex)
        o1_response = jnp.zeros((N), dtype=complex)

        for i in range(N):
            a = complex(inputs['o0'][i])
            b = complex(inputs['o1'][i])
            o0_response = o0_response.at[i].set(self.transmission * self.backward_wave.get())
            o1_response = o1_response.at[i].set(self.transmission * self.forward_wave.get())

            self.forward_wave.put(a)
            self.backward_wave.put(b)
        
        response = {
            "o0": o0_response,
            "o1": o1_response,
        }
        
        return response
