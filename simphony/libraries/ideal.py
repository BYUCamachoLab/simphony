# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Ideal models for common photonic components.

These are typically lossless and have no wavelength dependance.
"""

from typing import List, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    from simphony.utils import jax

    JAX_AVAILABLE = False

from simphony.context import CTX
from simphony.models import Model


class Coupler(Model):
    r"""2x2 photonic coupler model.

    .. image:: /_static/images/coupler.png
        :alt: coupler.png
        :align: center

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
    coupling : float, optional
        Coupling coefficient (0 <= coupling <= 1). Defaults to 0.5.
    phi : float, optional
        Phase shift between the two output ports (in radians). Defaults to pi/2.
    loss : float or tuple of floats, optional
        Total transmission of the component in dB (0 <= loss) assumed uniform
        loss across ports. If a tuple of 4 floats is given, the loss associated
        with each port is set individually.
    """
    onames = ["o0", "o1", "o2", "o3"]

    def __init__(
        self,
        coupling: float = 0.5,
        phi: float = jnp.pi / 2,
        loss: Union[float, Tuple[float]] = 0.0,
        name: str = None,
    ):
        super().__init__(name=name)
        if not 0 <= coupling <= 1:
            raise ValueError("coupling must be between 0 and 1")
        self.coupling = coupling
        # check if loss is iterable
        if isinstance(loss, tuple):
            if len(loss) != 4:
                raise ValueError("loss must be a single value or a tuple of 4 values")
            for l in loss:
                if not 0 <= l:
                    raise ValueError("loss must be greater than or equal to 0")
        elif isinstance(loss, float):
            if not 0 <= loss:
                raise ValueError("loss must be greater than 0")
        else:
            raise ValueError("loss must be a single value or a tuple of 4 values")
        self.loss = loss
        self.phi = phi

    def s_params(self, wl):
        try:
            L0, L1, L2, L3 = self.loss
            T0, T1, T2, T3 = 10 ** (-jnp.asarray([L0, L1, L2, L3]) / 10)
        except:
            T0 = T1 = T2 = T3 = (10 ** (self.loss / 10)) ** (1 / 2)
        t = jnp.sqrt(1 - self.coupling)
        r = jnp.sqrt(self.coupling)
        rp = jnp.conj(r)

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


# TODO: Check the Notes section for accuracy.
class Waveguide(Model):
    r"""Model of a 500 nm wide, 220 nm tall waveguide with a 1.55 um center
    wavelength.

    An ideal waveguide has transmission that is affected only by the loss of the
    waveguide and no reflection. The accumulated phase of the waveguide is
    affected by the effective index and the length of the waveguide.

    Parameters
    ----------
    length : float
        Length of the waveguide in microns.
    wl0 : float, optional
        Center wavelength of the waveguide in microns. Defaults to 1.55.
    neff : float, optional
        Effective index of the waveguide. If not set, reads the global effective
        index from the context.
    ng : float, optional
        Group index of the waveguide. If not set, reads the global group index
        from the context.
    loss : float, optional
        Loss of the waveguide in dB/micron (must be >= 0). If not set, reads the global loss
        from the context.

    Notes
    -----
    The effective index of the waveguide is calculated as:

    .. math::

        n_{\text{eff}} = n_g - \frac{\Delta \lambda}{\lambda_0} \frac{\partial n_g}{\partial \lambda}

    where :math:`n_g` is the group index, :math:`\Delta \lambda` is the
    wavelength difference between the center wavelength and the current
    wavelength, and :math:`\lambda_0` is the center wavelength.

    The transmission of the waveguide is calculated as:

    .. math::

        T = \exp \left( -\frac{2 \pi n_{\text{eff}} L}{\lambda} \right)

    where :math:`n_{\text{eff}}` is the effective index, :math:`L` is the length of the
    waveguide, and :math:`\lambda` is the current wavelength.

    The reflection of the waveguide is calculated as:

    .. math::

        R = \exp \left( -\frac{2 \pi n_{\text{eff}} L}{\lambda} \right) \exp \left( -\frac{2 \pi n_{\text{eff}} L}{\lambda_0} \right)

    where :math:`n_{\text{eff}}` is the effective index, :math:`L` is the length of the
    waveguide, and :math:`\lambda` is the current wavelength.

    The s-parameter matrix of the waveguide is calculated as:

    .. math::

        M = \begin{bmatrix}
                R & T \\
                T & R
            \end{bmatrix}
    """

    ocount = 2

    def __init__(
        self,
        length: float,
        wl0: float = 1.55,
        neff: float = None,
        ng: float = None,
        loss: float = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.length = length
        self.wl0 = wl0
        self.neff = neff
        self.ng = ng
        self.loss = loss

    def s_params(self, wl):
        global CTX

        _neff = self.neff or CTX.neff
        _ng = self.ng or CTX.ng
        _loss = self.loss if self.loss is not None else CTX.loss_db_cm / 100

        neff = _neff - (wl - self.wl0) * (_ng - _neff) / self.wl0
        amp = 10 ** (-_loss * self.length / 20)
        phase = 2 * jnp.pi * neff * self.length / wl
        s21 = amp * jnp.exp(-1j * phase)
        s12 = jnp.conj(s21)
        s11 = s22 = jnp.array([0] * len(wl))
        return jnp.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)


class PhaseShifter(Model):
    """Ideal phase shifter model.

    The ideal phase shifter has 2 ports ('o0', 'o1'), it acts as a waveguide
    with a set amount of phase shift.

    Parameters
    ----------
    phase : float
        Phase shift of the phase shifter in radians. Defaults to 0.
    loss : float
        Loss of the phase shifter in dB. Defaults to 0.
    name : str, optional
        Optional name for the component in the circuit.
    """

    onames = ["o0", "o1"]

    def __init__(self, phase: float = 0, loss: float = 0, name: str = None):
        super().__init__(name=name)
        self.phase = phase
        self.loss = loss

    def s_params(self, wl):
        s11 = s22 = jnp.array([0] * len(wl), dtype=jnp.complex64)
        s21 = jnp.array([10 ** (self.loss / 20) * jnp.exp(1j * self.phase)] * len(wl))
        s12 = jnp.conj(s21)
        return jnp.stack([s11, s12, s21, s22], axis=1).reshape(-1, 2, 2)


class Terminator(Model):
    """Ideal terminator model.

    The ideal terminator has 1 port ('o0'), it is essentially an ideal dump
    port. Useful for when you want to terminate unused ports.

    Parameters
    ----------
    onames : List[str]
        Name of the termination port. Defaults to ['o0'].
    """

    ocount = 1

    def s_params(self, wl: Union[float, np.ndarray]) -> np.ndarray:
        wl = np.asarray(wl).reshape(-1)
        s = np.zeros((len(wl), 1, 1), dtype=np.complex128)
        return s
