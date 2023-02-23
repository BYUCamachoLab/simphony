# MIT License
# -----------
#
# Copyright (c) 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt)
#
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
simphony
========

A Simulator for Photonic circuits
"""

import sys
import platform
import warnings
from types import SimpleNamespace


from .models_old import Model  # noqa: F401

if sys.version_info < (3, 8, 0):
    raise Exception(
        "Simphony requires Python 3.8+ (version "
        + platform.python_version()
        + " detected)."
    )

__version__ = "0.6.1"
__license__ = "MIT"
__project_url__ = "https://github.com/BYUCamachoLab/simphony"
__forum_url__ = "https://github.com/BYUCamachoLab/simphony/issues"
__trouble_url__ = __project_url__ + "/wiki/Troubleshooting-Guide"
__website_url__ = "https://camacholab.byu.edu/"


try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    def jit(func, *args, **kwargs):
        warnings.warn("Jax not available, cannot compile using 'jit'!")
        return func

    jax = SimpleNamespace(jit=jit)
    JAX_AVAILABLE = False
