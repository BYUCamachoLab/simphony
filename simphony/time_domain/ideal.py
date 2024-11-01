"""Ideal time-domain models."""
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from simphony.time_domain.time_system import TimeSystem
import simphony.libraries.ideal as fd
from simphony.utils import dict_to_matrix, mul_polar
import queue

class coupler(TimeSystem):
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

class waveguide(TimeSystem):
    def __init__(
        self,
        wl: ArrayLike | float = 1.55,
        wl0: float = 1.55,
        neff: float = 2.34,
        ng: float = 3.4,
        length: float = 10.0,
        loss: float = 0.0,
    ) -> None:
        super().__init__()
        pass
    
    def response():
        pass