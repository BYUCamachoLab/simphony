# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
This package contains handy functions useful across simphony submodules
and to the average user.
"""

import warnings
from types import SimpleNamespace

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    
    def jit(func, *args, **kwargs):
        """Mock "jit" version of a function. Warning is only raised once."""
        warnings.warn("Jax not available, cannot compile using 'jit'!")
        return func

    jax = SimpleNamespace(jit=jit)
    JAX_AVAILABLE = False
