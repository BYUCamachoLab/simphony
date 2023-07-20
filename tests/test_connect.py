import pytest
from simphony.connect import connect_s, vector_innerconnect_s, innerconnect_s

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    import numpy as np
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False


ArrSingleFreq_A = jnp.array([[[0, 0.707 + 0.707j], [0.707 + 0.707j, 0]]])

ArrSingleFreq_B = jnp.array([[[0, 0.707 + 0.707j], [0.707 + 0.707j, 0]]])

ArrSingleFreq_polar_A = jnp.array(
    [[[0, 1 * jnp.exp(1j * jnp.pi / 4)], [1 * jnp.exp(1j * jnp.pi / 4), 0]]]
)

ArrSingleFreq_polar_B = jnp.array([[[0, 0.707 + 0.707j], [0.707 + 0.707j, 0]]])

ArrMultipleFreq_A = jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, 0.5], [0.5, 0]],
        [[0, 0.5 * jnp.exp(1j * jnp.pi / 2)], [0.5 * jnp.exp(1j * jnp.pi / 2), 0]],
        [
            [0, 0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2)],
            [0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2), 0],
        ],
    ]
)

ArrMultipleFreq_B = jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, 0.5], [0.5, 0]],
        [[0, 0.5 * jnp.exp(1j * jnp.pi / 2)], [0.5 * jnp.exp(1j * jnp.pi / 2), 0]],
        [
            [0, 0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2)],
            [0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2), 0],
        ],
    ]
)

ArrMultipleFreq_ALong = jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, 0.5], [0.5, 0]],
        [[0, 0.5 * jnp.exp(1j * jnp.pi / 2)], [0.5 * jnp.exp(1j * jnp.pi / 2), 0]],
        [
            [0, 0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2)],
            [0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2), 0],
        ],
    ]
)

ArrMultipleFreq_BLong = jnp.array(
    [
        [[0, 1], [1, 0]],
        [[0, 0.5], [0.5, 0]],
        [[0, 0.5 * jnp.exp(1j * jnp.pi / 2)], [0.5 * jnp.exp(1j * jnp.pi / 2), 0]],
        [
            [0, 0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2)],
            [0.5 * jnp.exp(1j * 1.17 * jnp.pi / 2), 0],
        ],
    ]
)

SingleFreq = jnp.array([[[0.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]])
MultipleFreq = jnp.array(
    [
        [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        [
            [0.0 + 0.0j, 0.25 + 0.0j],
            [0.25 + 0.0j, 0.0 + 0.0j],
        ],
        [
            [0.0 + 0.0j, -0.25 - 2.1855694e-08j],
            [-0.25 - 2.1855694e-08j, 0.0 + 0.0j],
        ],
        [
            [0.0 + 0.0j, -0.2151855 - 1.2726036e-01j],
            [-0.2151855 - 1.2726036e-01j, 0.0 + 0.0j],
        ],
    ]
)


class TestConnect:
    def test_single_freq(self):
        Smatrix = connect_s(ArrSingleFreq_A, 0, ArrSingleFreq_B, 0)
        print("single connected:", Smatrix)
        assert jnp.allclose(Smatrix, SingleFreq, atol=1e-3, rtol=1e-3)

    def test_multiple_freq(self):
        Smatrix = connect_s(ArrMultipleFreq_A, 0, ArrMultipleFreq_A, 0)
        print("multiple connected:", Smatrix)
        assert jnp.allclose(Smatrix, MultipleFreq, atol=1e-3, rtol=1e-3)
