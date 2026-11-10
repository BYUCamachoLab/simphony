import jax
import jax.numpy as jnp
import jax.scipy.signal
from jax.typing import ArrayLike

from simphony.component.component import (
    BlockModeComponent,
    SampleModeComponent,
    SteadyStateComponent,
)
from simphony.component.port import Port
from simphony.signal.block_mode import BlockModeElectricalSignal
from simphony.signal.sample_mode import SampleModeElectricalSignal
from simphony.simulation.simulation import SimulationParameters


class VoltageFollower(
    SteadyStateComponent,
    SampleModeComponent,
    BlockModeComponent,
):
    """Pass an electrical signal from input port `e0` to output port `e1`.

    This helper is useful in directed Block mode netlists when an
    electrical signal needs an explicit through component. The Block
    mode response returns the input electrical signal object unchanged.
    """

    ports = [
        Port(
            name="e0",
            type="electrical",
            directionality="input",
        ),
        Port(
            name="e1",
            type="electrical",
            directionality="output",
        ),
    ]

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
    ):
        pass

    def steady_state(
        self,
        inputs: dict,
        simulation_parameters: SimulationParameters,
    ):
        outputs = {"e1": inputs["e0"]}
        return outputs

    def block_mode_response(self, input_signals: ArrayLike, simulation_parameters):
        return {"e1": input_signals["e0"]}

    def sample_mode_step(
        self, inputs: dict, state: jax.Array, simulation_state, simulation_parameters
    ):
        # TODO: Complete this to use the signal defined in settings
        return inputs, state

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])


class ElectricalFIRFilter(SampleModeComponent, BlockModeComponent):
    """Finite-impulse-response filter for an electrical signal.

    The filter reads the voltage on `e0` and writes

        y[n] = sum_k taps[k] * u[n - k + delay_compensation]

    on `e1`. The taps are in samples of the filter's own rate region, so behind
    an `Interpolator(factor=L)` one tap spans `dt / L` of the reference rate.
    That makes `Interpolator` + `ElectricalFIRFilter` a DAC model: a
    zero-order hold (`mode="hold"`) followed by the driver's low-pass response,
    or zero-stuffing (`mode="zeros"`) followed by a pulse-shaping filter.

    IIR responses (Bessel, Butterworth, ...) are used by passing their impulse
    response, truncated where it has decayed, e.g. from
    `scipy.signal.sosfilt(sos, unit_impulse(n))`.

    Parameters
    ----------
    taps:
        One-dimensional real or complex filter coefficients.
    delay_compensation:
        Number of samples by which the output is advanced to remove the
        filter's latency (block mode only; the end of the block is
        zero-padded). ``"group"`` uses ``(len(taps) - 1) // 2``, the group
        delay of a symmetric (linear-phase) filter. Sample mode is causal and
        accepts only ``0``.
    """

    ports = [
        Port(name="e0", type="electrical", directionality="input"),
        Port(name="e1", type="electrical", directionality="output"),
    ]

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        *,
        taps: ArrayLike = (1.0,),
        delay_compensation: int | str = 0,
    ):
        taps = jnp.atleast_1d(jnp.asarray(taps))
        if taps.ndim != 1 or taps.shape[0] == 0:
            raise ValueError(f"taps must be a non-empty 1-D array, got {taps.shape}")
        self.taps = taps
        if delay_compensation == "group":
            delay_compensation = (taps.shape[0] - 1) // 2
        if int(delay_compensation) != delay_compensation or delay_compensation < 0:
            raise ValueError(
                "delay_compensation must be a non-negative integer or 'group', got "
                f"{delay_compensation!r}"
            )
        self.delay_compensation = int(delay_compensation)

    def block_mode_response(self, input_signals, simulation_parameters):
        u = jnp.asarray(input_signals["e0"].voltage)
        n = u.shape[0]
        d = self.delay_compensation
        y = jax.scipy.signal.fftconvolve(u, self.taps.astype(u.dtype), mode="full")
        y = y[d : d + n]
        if y.shape[0] < n:
            y = jnp.concatenate([y, jnp.zeros(n - y.shape[0], dtype=y.dtype)])
        if not jnp.iscomplexobj(u) and not jnp.iscomplexobj(self.taps):
            y = jnp.real(y)
        return {"e1": BlockModeElectricalSignal(voltage=y)}

    def sample_mode_initial_state(self, simulation_parameters):
        if self.delay_compensation != 0:
            raise ValueError(
                "ElectricalFIRFilter: sample mode is causal; delay_compensation "
                f"must be 0, got {self.delay_compensation}"
            )
        # Last len(taps) inputs, newest first.
        return jnp.zeros(self.taps.shape[0], dtype=complex)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        u = jnp.asarray(inputs["e0"].voltage, dtype=complex).reshape(())
        state = jnp.concatenate([u[None], state[:-1]])
        y = jnp.sum(self.taps * state)
        if not (jnp.iscomplexobj(self.taps) or jnp.iscomplexobj(inputs["e0"].voltage)):
            y = y.real
        return {"e1": SampleModeElectricalSignal(voltage=y)}, state
