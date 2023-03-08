from typing import List, Union
from functools import partial

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
    r"""
    2x2 photonic coupler model.

    .. image:: /reference/images/coupler.png
        :alt: coupler.png

    The coupler has 2 inputs ('o0', 'o1') and 2 outputs ('o2', 'o3').
    The coupler has the following s-parameter matrix:

    .. math::
        M = \begin{bmatrix}
                0 & 0 & t \sqrt{T_0 T_2} & r^* \sqrt{T_0 T_3} \\
                0 & 0 & -r \sqrt{T_1 T_2} & t \sqrt{T_1 T_3} \\
                t \sqrt{T_0 T_2} & -r^* \sqrt{T_1 T_2} & 0 & 0 \\
                r \sqrt{T_0 T_3} & t \sqrt{T_1 T_3} & 0 & 0
            \end{bmatrix}

    where :math:`t = \sqrt{1 - \text{coupling}}` and :math:`r = 
    \sqrt{\text{coupling}}`.


    Parameters
    ----------
    coupling : float
        Coupling coefficient (0 <= coupling <= 1). Defaults to 0.5.
    phi : float
        Phase shift between the two output ports (in radians). Defaults to pi/2.
    loss : float or list of floats
        Loss of the component in dB (0 <= loss) assumed uniform loss across 
        ports. If a list of 4 floats is given, the loss associated with each 
        port is set individually.
    """
    # jit = False
    onames = ["o0", "o1", "o2", "o3"]

    def __init__(
        self,
        coupling: float = 0.5,
        phi: float = jnp.pi / 2,
        loss: Union[float, List[float]] = 0.0,
    ):
        self.coupling = coupling
        self.phi = phi
        self.loss = loss
        try:
            self.T0, self.T1, self.T2, self.T3 = 10 ** (loss / 10)
        except:
            self.T0 = self.T1 = self.T2 = self.T3 = 10 ** (loss / 10) ** (1 / 4)
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
            ], dtype=jnp.float32
        )
        # fmt: on

        # repeat smatrix for each wavelength since its frequency independent
        return jnp.stack([smatrix] * len(wl), axis=0)


class Waveguide(Model):
    ocount = 2
    ecount = 2

    def __init__(
        self,
        length: float = 15.0,
        loss: float = 0.0,
        neff: float = 2.34,
        ng: float = 3.4,
        wl0: float = 1.55,
    ):
        self.length = length
        self.loss = loss
        self.neff = neff
        self.ng = ng
        self.wl0 = wl0

    def s_params(self, wl):
        neff = self.neff - (wl - self.wl0) * (self.ng - self.neff) / self.wl0
        amp = 10 ** (-self.loss * self.length / 20)
        phase = 2 * jnp.pi * neff * self.length / wl
        s21 = amp * jnp.exp(-1j * phase)
        s12 = jnp.conj(s21)
        s11 = s22 = jnp.array([0] * len(wl))
        return jnp.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)


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
    print(coupler.s_params(jnp.array([0, 0])))
