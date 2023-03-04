from typing import List, Union

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False

from simphony.models import Model
from simphony.context import CTX


class Coupler(Model):
    onames = ["o1", "o2", "o3", "o4"]

    def __init__(
        self,
        coupling: float = 0.5,
        phi: float = jnp.pi / 2,
        loss: Union[float, List[float]] = 0.0,
    ):
        self.coupling = coupling
        self.phi = phi
        if hasattr(loss, "__iter__"):
            self.T0, self.T1, self.T2, self.T3 = loss
        else:
            self.T0 = self.T1 = self.T2 = self.T3 = loss ** (1 / 4)
        self.t = jnp.sqrt(1 - self.coupling)
        self.r = jnp.sqrt(self.coupling)
        self.rp = jnp.conj(self.r)

    def s_params(self, wl):
        t, r, rp = self.t, self.r, self.rp
        T0, T1, T2, T3 = self.T0, self.T1, self.T2, self.T3
        # fmt: off
        smatrix = jnp.array(
            [
                [0, 0, t * jnp.sqrt(T0*T2),  rp * jnp.sqrt(T0*T3)],
                [0, 0, -r * jnp.sqrt(T1*T2),  t * jnp.sqrt(T1*T3)],
                [t * jnp.sqrt(T0*T2), -rp * jnp.sqrt(T1*T2), 0, 0],
                [r * jnp.sqrt(T0*T3), t * jnp.sqrt(T1*T3), 0, 0]
            ]
        )
        # fmt: on

        # repeat smatrix for each wavelength since its frequency independent
        return jnp.repeat(smatrix, len(wl), axis=0)


class Waveguide(Model):
    ocount = 2
    ecount = 2

    def __init__(self, a):
        self.a = a

    def s_params(self, wl):
        # Fake value
        print(f"cache miss ({self.a})")
        return 1j * self.a


class Heater(Model):
    jit = False

    def __init__(self, onames=["o0", "o1"]):
        self.onames = onames

    def s_params(self, wl):
        # Fake value
        pass


if __name__ == "__main__":
    # Create a coupler
    coupler = Coupler()

    coupler.s_params([1.5, 1.6])