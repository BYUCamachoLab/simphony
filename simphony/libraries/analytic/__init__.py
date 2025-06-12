# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Simphony models compatible with Simphony.Circuit.

This package contains parameterized models of basic PIC components,
compatible with Time-domain and Frequency-domain Simphony.

Usage:

.. code-block:: python

    from simphony.libraries import analytic

    wg = analytic.Waveguide()
"""

from simphony.libraries.analytic.couplers import (
    star_coupler,
)
from simphony.libraries.analytic.modulators import (
    MachZehnderModulator,
    PhaseModulator,
)
from simphony.libraries.analytic.s_parameters import (
    optical_s_parameter,
)
from simphony.libraries.analytic.sources import (
    PRNG,
    CWLaser,
    VoltageSource,
)
from simphony.libraries.analytic.waveguides import (
    Fiber,
    GRINFiber,
    Waveguide,
)

__all__ = [
    "Waveguide",
    "Fiber",
    "GRINFiber",
    "CWLaser",
    "MachZehnderModulator",
    "PhaseModulator",
    "VoltageSource",
    "PRNG",
    "star_coupler",
    "optical_s_parameter",
]
