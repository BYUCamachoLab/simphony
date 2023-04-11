import pytest
from cmath import exp
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
    
    
ArrSingleFreq_A = jnp.array(
    [[[0, .707+.707j],
      [.707+.707j, 0]]
     ])

ArrSingleFreq_B = jnp.array(
    [[[0, .707+.707j],
     [.707+.707j, 0]]
     ])

ArrMultipleFreq_A = jnp.array(
    [[[0, 1],
      [1, 0]],
     [[0, .5],
      [.5, 0]],
     [[0, .5*exp(1j*np.pi/2)],
      [.5*exp(1j*jnp.pi/2), 0]],
     [[0, .5*exp(1j*1.17*jnp.pi/2)],
      [.5*exp(1j*1.17*jnp.pi/2), 0]]
     ])

ArrMultipleFreq_B = jnp.array(
    [[[0, 1],
     [1, 0]],
     [[0, .5],
      [.5, 0]],
     [[0, .5*exp(1j*jnp.pi/2)],
      [.5*exp(1j*jnp.pi/2), 0]],
     [[0, .5*exp(1j*1.17*jnp.pi/2)],
      [.5*exp(1j*1.17*jnp.pi/2), 0]]
     ])

ArrMultipleFreq_ALong = jnp.array(
    [[[0, 1],
      [1, 0]],
     [[0, .5],
      [.5, 0]],
     [[0, .5*exp(1j*np.pi/2)],
      [.5*exp(1j*jnp.pi/2), 0]],
     [[0, .5*exp(1j*1.17*jnp.pi/2)],
      [.5*exp(1j*1.17*jnp.pi/2), 0]]
     ])

ArrMultipleFreq_BLong = jnp.array(
    [[[0, 1],
     [1, 0]],
     [[0, .5],
      [.5, 0]],
     [[0, .5*exp(1j*jnp.pi/2)],
      [.5*exp(1j*jnp.pi/2), 0]],
     [[0, .5*exp(1j*1.17*jnp.pi/2)],
      [.5*exp(1j*1.17*jnp.pi/2), 0]]
     ])

SingleFreq = jnp.array(
    [[[0.+0.j, 0.+1.j],
    [0.+1.j, 0.+0.j]]])


class TestConnect:
    def test_single_freq(self):
        Smatrix = connect_s(ArrSingleFreq_A, 0, ArrSingleFreq_B, 0)
        # print("single connected:", Smatrix)
        assert jnp.allclose(Smatrix,SingleFreq,atol = 1e-3, rtol = 1e-3)
        
    def test_multiple_freq(self):
        Smatrix = connect_s(ArrMultipleFreq_A, 0, ArrMultipleFreq_A, 0)
        print("multiple connected:", Smatrix)
        assert Smatrix == SingleFreq