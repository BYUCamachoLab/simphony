"""Rate-changing components: decimators and interpolators.

Both work on any signal type that declares `_data_fields` (see
`simphony.signal.utils`), so the same component resamples optical, electrical,
logic or vector signals. They follow the conventions of Simulink's Downsample
and Upsample blocks:

* `Decimator(factor=M, offset=D)`: y[m] = u[m*M + D], 0 <= D < M
  ("sample offset").
* `Interpolator(factor=L, offset=D)`: y[n*L + D] = u[n]; the other L-1
  samples are zeros (`mode="zeros"`) or repeat the latest sample
  (`mode="hold"`, the default; a zero-order hold whose transitions are delayed
  by D output samples).

Block mode implements these equations exactly. In sample mode, every edge adds
one sample of latency at the producing rate, as in single-rate circuits, so
the decimator outputs u[m*M + D - 1] and the interpolator outputs the input
sample before the one in the block-mode equation.

Neither component filters. Put an anti-aliasing filter before a decimator, or
an anti-imaging filter after a zero-stuffing interpolator, when that matters.
Backward-travelling waves are absorbed (not resampled).
"""

from fractions import Fraction

import jax.numpy as jnp

from simphony.component.component import (
    BlockModeComponent,
    RateChanger,
    SampleModeComponent,
)
from simphony.component.port import Port
from simphony.signal.utils import (
    data_fields,
    decimate_block,
    upsample_block,
    zeros_like_signal,
)


def _check_factor_offset(factor, offset):
    if int(factor) != factor or factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    if int(offset) != offset or not 0 <= offset < factor:
        raise ValueError(
            f"offset must be an integer in [0, {factor - 1}], got {offset!r}"
        )
    return int(factor), int(offset)


class Decimator(RateChanger, SampleModeComponent, BlockModeComponent):
    """Keep every `factor`-th sample, starting at sample `offset`.

    The output region runs at 1/`factor` of the input rate, and its first
    sample is input sample `offset`. In sample mode the decimator (and every
    component downstream of it) is only evaluated on those samples.
    """

    ports = [
        Port(name="in", type="any", directionality="input"),
        Port(name="out", type="any", directionality="output"),
    ]

    def __init__(self, simulation_parameters, *, factor: int = 2, offset: int = 0):
        self.factor, self.offset = _check_factor_offset(factor, offset)
        self.rate_ratio = Fraction(1, self.factor)

    def output_phase(self, input_phase, input_period):
        return input_phase + self.offset * input_period

    def block_mode_response(self, inputs, simulation_parameters):
        return {"out": decimate_block(inputs["in"], self.factor, self.offset)}

    def sample_mode_output_template(self, port, simulation_parameters):
        return None  # same signal type and shape as the input

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        return {"out": inputs["in"]}, state


class Interpolator(RateChanger, SampleModeComponent, BlockModeComponent):
    """Raise the sample rate by `factor`.

    `mode="hold"` (default) repeats each input sample `factor` times;
    `mode="zeros"` inserts `factor - 1` zeros. `offset` delays where each input
    sample lands by that many output samples, as in Simulink's Upsample.
    """

    ports = [
        Port(name="in", type="any", directionality="input"),
        Port(name="out", type="any", directionality="output"),
    ]

    def __init__(
        self,
        simulation_parameters,
        *,
        factor: int = 2,
        offset: int = 0,
        mode: str = "hold",
    ):
        self.factor, self.offset = _check_factor_offset(factor, offset)
        if mode not in ("hold", "zeros"):
            raise ValueError(f"mode must be 'hold' or 'zeros', got {mode!r}")
        self.mode = mode
        self.rate_ratio = Fraction(self.factor)

    def block_mode_response(self, inputs, simulation_parameters):
        return {
            "out": upsample_block(inputs["in"], self.factor, self.offset, self.mode)
        }

    def sample_mode_output_template(self, port, simulation_parameters):
        return None  # same signal type and shape as the input

    def sample_mode_initial_state(self, simulation_parameters):
        # (output-sample counter, latched input, held output). The simulator
        # sets `_input_template` (a zero signal shaped like the input).
        zero = zeros_like_signal(self._input_template)
        return (jnp.asarray(0), zero, zero)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        counter, latched, held = state
        phase = counter % self.factor
        incoming = inputs["in"]
        # Latch a new input sample at the start of every input period.
        latched = _select(phase == 0, incoming, latched)
        # The latched sample becomes the output at offset D.
        held = _select(phase == self.offset, latched, held)
        if self.mode == "hold":
            out = held
        else:
            out = _select(phase == self.offset, latched, zeros_like_signal(latched))
        return {"out": out}, (counter + 1, latched, held)


def _select(predicate, a, b):
    """Elementwise `a if predicate else b` over two signals of one type."""
    return a.replace(
        **{
            f: jnp.where(
                predicate, jnp.asarray(getattr(a, f)), jnp.asarray(getattr(b, f))
            )
            for f in data_fields(a)
        }
    )
