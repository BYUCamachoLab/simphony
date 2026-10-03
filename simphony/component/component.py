###### Necessary ######
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.block_mode import (
        BlockModeSimulationParameters,
    )  # Only imported for type checking
    from simulation.sample_mode import SampleModeSimulationParameters
    from simulation.simulation import SimulationMode

import inspect
from fractions import Fraction
from typing import Tuple

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike
from scipy.constants import speed_of_light

from simphony.signal.block_mode import BlockModeElectricalSignal, BlockModeOpticalSignal
from simphony.signal.sample_mode import (
    SampleModeElectricalSignal,
    SampleModeLogicSignal,
    SampleModeOpticalSignal,
    SampleModeTemperatureSignal,
)


class Signal:  ## TODO: Make an actual base class
    ...


class _hybridmethod:
    """Descriptor binding a method to the instance when accessed from one,
    and to the class otherwise."""

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        target = cls if obj is None else obj
        return self.func.__get__(target, type(target))


class Component:
    """Base class for objects that can be placed in a Simphony circuit.

    Concrete component classes define a class-level `ports` list
    containing `Port` objects. Simulator-specific mixins such as
    `BlockModeComponent`, `SampleModeComponent`, and
    `SParameterComponent` then define the response methods that a
    simulator is allowed to call.
    """

    def __repr__(self):
        return f"<{type(self).__name__} (Component obj)>"

    @_hybridmethod
    def _create_port_lookup_table(obj):
        """Build internal lookup tables from the component's declared ports.

        Called on a class, the tables are built from (and stored on) the
        class. Called on an instance, they are built from `obj.ports`, which
        may be an instance attribute, and stored on the instance only. This
        lets instances of the same component class carry different port
        directionalities.

        Bidirectional ports are included in both the input and output
        lookup tables because they can participate on either side
        depending on the simulation mode and netlist orientation.
        """
        obj._port_lookup_table = {p.name: p for p in obj.ports}
        obj._input_port_lookup_table = {
            p.name: p
            for p in obj.ports
            if p.directionality in ("input", "bidirectional")
        }
        obj._output_port_lookup_table = {
            p.name: p
            for p in obj.ports
            if p.directionality in ("output", "bidirectional")
        }

    def __init__(
        self,
        simulation_mode: SimulationMode,
        **kwargs,
    ):
        raise ValueError("Component is a base class")


class SteadyStateComponent(Component):
    """Mixin for components that can compute a static operating point.

    Steady-state responses are commonly used to provide bias values for
    frequency-domain or S-parameter simulations. Implementations receive
    a dictionary of input signals keyed by port name and return output
    signals in the same style.
    """

    delay_compensation = 0

    def steady_state(self, inputs: dict) -> dict:
        """Compute steady-state output signals.

        Parameters
        ----------
        inputs:
            Mapping from input port name to steady-state signal object.

        Returns
        -------
        dict
            Mapping from output port name to steady-state signal object.
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )


class BlockModeComponent(Component):
    """Base class for components that process an entire time block at once.

    User-defined Block mode components should declare a class-level
    `ports` list and implement `block_mode_response`. The simulator
    supplies a dictionary of signals keyed by port name, and the method
    returns a dictionary of signals keyed by port name.

    Port directionality defines which way is "forward" for the simulator,
    but it does not forbid backward-travelling waves on optical ports:

    - An entry in the returned dict for an **output** port is a
      forward-travelling wave leaving the component.
    - An entry in the returned dict for an **input** port is a
      backward-travelling wave leaving the component (for example a
      reflection).
    - An entry in `input_signals` for an **output** port is a
      backward-travelling wave arriving at the component. These only
      appear during the optional backward pass of the Block mode simulator
      (`BlockModeSimulationParameters.backward_pass`).

    Any wave that is missing from either dict is treated as zero, so
    components that do not model backward-travelling waves simply ignore
    them (they absorb backward waves).
    """

    def block_mode_response(
        self,
        input_signals: ArrayLike,
        simulation_parameters: BlockModeSimulationParameters,
    ):
        """Compute output signals for one full Block mode time block.

        Parameters
        ----------
        input_signals:
            Mapping from port name to a block signal object. Optical ports
            receive `BlockModeOpticalSignal`; electrical ports receive
            `BlockModeElectricalSignal`. Missing input ports are filled with
            zero signals; entries for output ports are backward-travelling
            waves (see the class docstring).
        simulation_parameters:
            Shared Block mode parameters defining the time grid, wavelengths,
            and modes.

        Returns
        -------
        dict
            Mapping from port name to block signal object. Entries for input
            ports are backward-travelling waves.
        """
        raise NotImplementedError

    def _block_mode_response(self, input_signals, simulation_parameters):
        # Rate changers receive their output-domain parameters; their inputs
        # live in the input domain (`input_parameters`, set by the simulator).
        input_parameters = getattr(self, "input_parameters", simulation_parameters)
        for port_name, port in self._input_port_lookup_table.items():
            if port_name not in input_signals:
                input_signals[port_name] = self._default_input_signal(
                    input_parameters, port.type
                )
        outputs = self.block_mode_response(input_signals, simulation_parameters)

        baseband_wls = simulation_parameters.optical_baseband_wavelengths
        time_steps = (
            jnp.arange(simulation_parameters.num_time_steps) * simulation_parameters.dt
            + simulation_parameters.time_offset
        )

        for port, signal in outputs.items():
            if not isinstance(signal, BlockModeOpticalSignal):
                continue

            amplitude = signal.amplitude
            wavelengths = signal.wavelength

            distances = jnp.abs(baseband_wls[:, None] - wavelengths[None, :])
            closest_idx = jnp.argmin(distances, axis=0)

            f_diff = (
                speed_of_light / wavelengths
                - speed_of_light / baseband_wls[closest_idx]
            )

            phase = jnp.exp(-1j * 2 * jnp.pi * time_steps[:, None] * f_diff[None, :])

            new_amplitude = jnp.zeros(
                (amplitude.shape[0], baseband_wls.shape[0], amplitude.shape[2]),
                dtype=complex,
            )

            new_amplitude = new_amplitude.at[:, closest_idx, :].add(
                amplitude * phase[:, :, None]
            )

            outputs[port] = BlockModeOpticalSignal(
                amplitude=new_amplitude,
                wavelength=baseband_wls,
            )

        return outputs

    def _default_input_signal(self, simulation_parameters, port_type):
        if port_type == "optical":
            wl = simulation_parameters.optical_baseband_wavelengths
            T = simulation_parameters.num_time_steps
            L = wl.shape[0]
            M = len(simulation_parameters.mode_identifiers)
            amplitude = jnp.zeros((T, L, M), dtype=complex)
            return BlockModeOpticalSignal(amplitude=amplitude, wavelength=wl)
        elif port_type == "electrical":
            T = simulation_parameters.num_time_steps
            voltage = jnp.zeros((T,), dtype=float)
            return BlockModeElectricalSignal(voltage=voltage)
        elif port_type == "logic":
            raise NotImplementedError  # TODO: Complete this function for more port types
        else:
            raise NotImplementedError(
                f"Default signal not specified for ports of type {port_type}"
            )  # TODO: Complete this function for more port types

    # TODO: Decide whether it is worth it to implement this function
    # def _default_output_signal(simulation_parameters, port_type):
    #     pass


class SampleModeComponent(Component):
    """Mixin for components that advance one simulation sample at a time.

    Sample mode components keep explicit state between time steps. The
    simulator first calls `sample_mode_initial_state`, then repeatedly
    calls `sample_mode_step` with the current inputs, component state,
    global simulation state, and simulation parameters.
    """

    def sample_mode_initial_state(
        self, simulation_parameters: SampleModeSimulationParameters
    ):
        """Return the component's initial sample-mode state.

        Components without internal memory can keep the default zero
        state.
        """
        return 0

    def sample_mode_step(
        self,
        inputs: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ) -> Tuple[dict[str, Signal], jax.Array]:
        """Compute one sample of output and the next component state.

        Returns
        -------
        tuple[dict, jax.Array]
            Output signals keyed by port name, followed by the updated internal
            state.
        """
        raise NotImplementedError

    def sample_mode_output_template(self, port, simulation_parameters):
        """Return a zero-valued signal with the shape this port emits.

        The sample-mode simulator uses it as the port's output before the
        component first fires. The default covers optical, electrical, logic
        and temperature ports; components with other port types (e.g. `"vector"`) must
        override it. Return `None` to inherit the template of the signal
        feeding the component (used by rate changers).
        """
        if port.type == "optical":
            wl = simulation_parameters.optical_baseband_wavelengths
            M = len(simulation_parameters.mode_identifiers)
            return SampleModeOpticalSignal(
                jnp.zeros((wl.shape[0], M), dtype=complex), wl
            )
        if port.type == "electrical":
            return SampleModeElectricalSignal(0.0)
        if port.type == "logic":
            return SampleModeLogicSignal(0)
        if port.type == "temperature":
            return SampleModeTemperatureSignal(0.0)
        raise NotImplementedError(
            f"{type(self).__name__} must implement sample_mode_output_template "
            f"for port {port.name!r} of type {port.type!r}"
        )

    def _sample_mode_initial_state(
        self, simulation_parameters: SampleModeSimulationParameters
    ):
        _initial_state = (
            0,
            self.sample_mode_initial_state(simulation_parameters=simulation_parameters),
        )
        self._output_optical_port_names = [
            p.name
            for p in self._output_port_lookup_table.values()
            if (p.directionality == "bidirectional" or p.directionality == "output")
            and p.type == "optical"
        ]
        return _initial_state

    # @partial(jax.jit, static_argnums=(0,))
    def _sample_mode_step(
        self,
        inputs: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        time_step = state[0]
        internal_state = state[1]

        f_s = 1 / simulation_parameters.dt

        outputs, output_state = self.sample_mode_step(
            inputs, internal_state, simulation_state, simulation_parameters
        )

        # Convert all inputs to the frequency channels in the simulator
        baseband_wls = simulation_parameters.optical_baseband_wavelengths
        for port_name in self._output_optical_port_names:
            signal = outputs[port_name]
            amplitude = signal.amplitude
            wavelength = signal.wavelength
            if wavelength is baseband_wls:
                continue

            dists = jnp.abs(baseband_wls[:, None] - wavelength[None, :])
            closest_idx = jnp.argmin(dists, axis=0)

            f_diff = (
                speed_of_light / wavelength - speed_of_light / baseband_wls[closest_idx]
            )

            new_amplitude = jnp.zeros(
                (baseband_wls.shape[0], amplitude.shape[1]), dtype=complex
            )
            new_amplitude = new_amplitude.at[closest_idx].add(
                amplitude
                * jnp.exp(
                    -1j
                    * 2
                    * jnp.pi
                    * f_diff[:, None]
                    * (time_step / f_s + simulation_parameters.time_offset)
                )
            )
            outputs[port_name] = signal.replace(
                amplitude=new_amplitude, wavelength=baseband_wls
            )

        return outputs, (time_step + 1, output_state)


class RateChanger(Component):
    """Mixin for components whose output ports run at a different sample rate
    than their input ports (decimators, interpolators, ...).

    `rate_ratio` is (output sample rate) / (input sample rate), an exact
    `Fraction`. Input ports form the component's "in" rate domain and output
    ports its "out" domain; `phase_shift` is the delay, in input or output
    samples as documented by the subclass, that the component adds to the
    sampling instants of the "out" domain (e.g. a decimator's sample offset).

    Multirate simulators give a rate changer the local simulation parameters
    of its *output* domain; `input_parameters` holds those of its input domain.
    """

    rate_ratio: Fraction = Fraction(1)

    def output_phase(self, input_phase, input_period):
        """Phase (time of the first output sample, in units of the reference
        dt) of the output domain, given the input domain's phase and period."""
        return input_phase


class SParameterComponent(Component):
    """Mixin for components described by wavelength-dependent S-parameters.

    This is only the interface used by `SParameterSimulation` ("this
    component can return an S-dict"). Components built from SAX models use
    the concrete `simphony.libraries.ideal.s_parameters.SParameterElement`,
    which additionally implements the sample-mode and block-mode responses.

    Optical ports are treated as scattering ports by default. Other
    signal types are typically bias ports whose steady-state values
    modify the returned S-parameter dictionary.
    """

    def s_parameters(
        self,
        inputs: dict,
        wl: ArrayLike = 1.55e-6,
    ):
        """Return the component S-parameter dictionary at wavelength `wl`.

        Parameters
        ----------
        inputs:
            Bias or control signals keyed by port name.
        wl:
            Wavelength or wavelength array, in meters.

        Returns
        -------
        sax.SDict
            Mapping from `(output_port, input_port)` to complex transmission.
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )

    def s_parameter_get_bias_ports(
        self,
    ):
        """Return ports whose steady-state inputs affect `s_parameters`.

        Bias ports receive steady-state signal values before the
        S-parameter response is evaluated.
        """
        return []

    def _s_parameter_get_bias_ports(
        self,
    ):
        """Bias ports are ports that recieve a steady state signal which in
        some way modify the s-dict of an SParameterComponent."""
        return self.s_parameter_get_bias_ports()


class GaussianProcessComponent(Component):
    """Base class for components that participate in GaussianProcessSimulation.

    Designers implement `gaussian_process_mode_response`; the simulator
    calls `_gaussian_process_mode_response` (the wrapper).
    """

    def gaussian_process_mode_response(
        self, inputs: dict, simulation_parameters
    ) -> dict:
        """Return a dict of GaussianProcessOpticalSignal for each output
        port."""
        raise NotImplementedError

    def _gaussian_process_mode_response(
        self, inputs: dict, simulation_parameters
    ) -> dict:
        for port_name, port in self._input_port_lookup_table.items():
            inputs.setdefault(
                port_name,
                self._default_gp_input_signal(simulation_parameters, port.type),
            )
        return self.gaussian_process_mode_response(inputs, simulation_parameters)

    def _default_gp_input_signal(self, simulation_parameters, port_type: str):
        from simphony.signal.gaussian_process import GaussianProcessOpticalSignal

        if port_type == "optical":
            wl = simulation_parameters.optical_baseband_wavelengths
            T = simulation_parameters.num_time_steps
            L = wl.shape[0]
            M = len(simulation_parameters.mode_identifiers)
            return GaussianProcessOpticalSignal(
                mean_amplitude=jnp.zeros((T, L, M), dtype=complex),
                covariance=jnp.zeros((L, T, T, M, M), dtype=complex),
                wavelength=wl,
            )
        raise NotImplementedError(
            f"No default GP signal defined for port type '{port_type}'"
        )
