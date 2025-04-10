from simphony.libraries import ideal

import jax.numpy as jnp

from itertools import product


class TestCoupler:
    def test_instantiable(self):
        # Test accross valid parameters
        params = product(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # coupling
            [0.0, 0.1, 1.0, 10.0],  # loss
            [0.0, jnp.pi / 2, jnp.pi],  # phi
        )
        for c, l, p in params:
            result = ideal.coupler(coupling=c, loss=l, phi=p)
            assert isinstance(
                result, dict
            ), f"Failed for coupling={c}, loss={l}, phi={p}"


class TestWaveguide:
    def test_instantiable(self):
        # Test accross valid parameters
        params = product(
            [1.5, 1.55, 1.6],  # wl0
            [2.0, 2.34, 3.0],  # neff
            [3.0, 3.4, 4.0],  # ng
            [5.0, 10.0, 15.0],  # length
            [0.0, 0.1, 1.0, 10.0],  # loss
        )
        for wo, ne, ng, l, lo in params:
            result = ideal.waveguide(wl0=wo, neff=ne, ng=ng, length=l, loss=lo)
            assert isinstance(
                result, dict
            ), f"Failed w/ wl0={wo}, neff={ne}, ng={ng}, length={l}, loss={lo}"

        # Test accross various wavelengths
        wavelengths = [1.5, 1.55, 1.6]
        for wl in wavelengths:
            result = ideal.waveguide(wl=wl)
            assert isinstance(result, dict), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wavelengths = jnp.array([1.5, 1.55, 1.6])
        result = ideal.waveguide(wl=wavelengths)
        assert isinstance(result, dict), f"Failed for wavelength array={wavelengths}"
