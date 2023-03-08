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
    r"""
    2x2 photonic coupler model.

    .. image:: /reference/images/coupler.png
        :alt: coupler.png

    The coupler has 2 inputs ('o1', 'o2') and 2 outputs ('o3', 'o4').
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
            self.T0, self.T1, self.T2, self.T3 = 10**(loss/10)
        else:
            self.T0 = self.T1 = self.T2 = self.T3 = 10**(loss/10) ** (1 / 4)
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
        return jnp.stack([smatrix]*len(wl), axis=0)


class Waveguide(Model):
    """
    Model of a 500 nm wide, 220 nm tall waveguide with a 1.55 um center wavelength.

    An ideal waveguide has transmission that is affected only by the loss of the
    waveguide and no reflection. The accumulated phase of the waveguide is
    affected by the effective index and the length of the waveguide.

    Parameters
    ----------
    length : float
        Length of the waveguide in microns.
    wl0 : float
        Center wavelength of the waveguide in microns. Defaults to 1.55.
    neff : float
        Effective index of the waveguide. If not set, reads the global effective
        index from the context.
    ng : float
        Group index of the waveguide. If not set, reads the global group index
        from the context.
    loss : float
        Loss of the waveguide in dB/micron. If not set, reads the global loss
        from the context.

    TODO: Check the following note for accuracy.

    Notes
    -----
    The effective index of the waveguide is calculated as:

    .. math::
        n_{eff} = n_g - \frac{\Delta \lambda}{\lambda_0} \frac{\partial n_g}{\partial \lambda}

    where :math:`n_g` is the group index, :math:`\Delta \lambda` is the
    wavelength difference between the center wavelength and the current
    wavelength, and :math:`\lambda_0` is the center wavelength.

    The transmission of the waveguide is calculated as:

    .. math::
        T = \exp(-\frac{2 \pi n_{eff} L}{\lambda})

    where :math:`n_{eff}` is the effective index, :math:`L` is the length of the
    waveguide, and :math:`\lambda` is the current wavelength.

    The reflection of the waveguide is calculated as:

    .. math::
        R = \exp(-\frac{2 \pi n_{eff} L}{\lambda}) \exp(-\frac{2 \pi n_{eff} L}{\lambda_0})

    where :math:`n_{eff}` is the effective index, :math:`L` is the length of the
    waveguide, and :math:`\lambda` is the current wavelength.

    The s-parameter matrix of the waveguide is calculated as:

    .. math::
        M = \begin{bmatrix}
                T & R \\
                R & T
            \end{bmatrix}
    """
    ocount = 2

    def __init__(self, length, wl0=1.55, neff=None, ng=None, loss=None):
        self.length = length
        self.wl0 = wl0
        self.neff = neff
        self.ng = ng
        self.loss = loss

    def s_params(self, wl):
        global CTX

        neff = self.neff or CTX.neff
        ng = self.ng or CTX.ng
        loss = self.loss or CTX.loss

        dwl = wl - self.wl0
        dneff_dwl = (ng - neff) / self.wl0
        neff = neff - dwl * dneff_dwl
        phase = 2 * jnp.pi * neff * self.length / wl
        amplitude = jnp.asarray(10 ** (-loss * self.length / 20), dtype=complex)
        transmission =  amplitude * jnp.exp(1j * phase)
        return jnp.array(
            [
                [transmission, 0],
                [0, transmission],
            ]
        )


class Heater(Model):
    def __init__(self, onames=["o0", "o1"]):
        self.onames = onames

    def s_params(self, wl):
        # Fake value
        pass


if __name__ == "__main__":
    # Create a coupler
    coupler = Coupler()
    print(coupler.s_params(jnp.array([0])))