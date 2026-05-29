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
from typing import Tuple

import jax
import jax.numpy as jnp

# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
# import gravis as gv
# import sax
from jax.typing import ArrayLike
from scipy.constants import speed_of_light

from simphony.signal.block_mode import BlockModeElectricalSignal, BlockModeOpticalSignal

#### Include First ####

# from dataclasses import replace

# from simphony.circuit.port import Port


# from sax.saxtypes import Model as SaxModel
# from simphony.time_domain.pole_residue_model import IIRModelBaseband
# from simphony.simulation.simulation import SimulationParameters

# from simphony.utils import dict_to_matrix
# from simphony.simulation import SampleModeSimulationParameters
# from copy import deepcopy
# from functools import partial
# from simphony.signals import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal

# from scipy.signal import butter, tf2ss, StateSpace, firwin, freqz, group_delay, cheby1, bessel
# from scipy.signal.windows import tukey
# from control import balred, ss
# import matplotlib.pyplot as plt
# from simphony.time_domain.vector_fitting.z_domain import optimize_order_vector_fitting_discrete, pole_residue_response_discrete, state_space_discrete

# from simphony.utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph
# from copy import deepcopy
# from simphony.signals import    steady_state_optical_signal, \
#                                 sample_mode_electrical_signal, \
#                                 sample_mode_optical_signal, \
#                                 sample_mode_logic_signal, \
#                                 block_mode_optical_signal, \
#                                 block_mode_electrical_signal, \
#                                 block_mode_logic_signal, \
#                                 complete_steady_state_inputs, \
#                                 complete_sample_mode_inputs
# from simphony.signal.block_mode import BlockModeElectricalSignal, BlockModeLogicSignal, BlockModeOpticalSignal
# from simphony.signal.steady_state import SteadyStateOpticalSignal


# from simphony.utils import dict_to_matrix
# from jax.scipy.special import i0
# from scipy.special import lambertw

# from scipy.signal.windows import kaiser_bessel_derived

# def line_of_best_fit_m(x, y):
#     x_mean = jnp.mean(x[:, None, None], axis=0)
#     y_mean = jnp.mean(y, axis=0)
#     cov = jnp.mean((x[:, None, None]-x_mean)*(y - y_mean), axis=0)
#     var = jnp.mean((x[:, None, None]-x_mean)**2, axis=0)
#     slope = cov/var
#     intercept = y_mean - slope * x_mean
#     return slope, intercept

# def _extension_up(m, b, x, y_initial, y_final):
#     k = m*(y_final - y_initial)

#     x_ext = x - x[0]
#     y_ext = y_final + (y_initial - y_final)*jnp.exp(-k*x)
#     return y_ext

# def extend_down(m, b, y_f=0.0, N=500):
#     pass

# def extend(x, y, x_min, x_max, alpha=1e11):
#     dy = y[-1] - y[-2]
#     dx = x[-1] - x[-2]
#     m = dy/dx
#     b = y[-1] - m*x[-1]

#     for i in range(y.shape[1]):
#         for j in range(y.shape[2]):
#             if m[i, j] > 0:
#                 y_initial = y[-1, i, j]
#                 p = 1 - jnp.exp(-alpha*m[i, j])
#                 y_final = y_initial + p*(1-y_initial)
#                 x_extension = jnp.arange(x[-1]+dx, x_max, dx)
#                 y_extension = _extension_up(m[i, j], b[i, j], x_extension, y_initial, y_final)

#                 x_extended = jnp.concatenate([x, x_extension])
#                 y_extended = jnp.concatenate([y[:, i, j], y_extension])
#                 # plt.plot(x_extended, y_extended)
#                 # plt.plot(x, y[:, i, j])
#                 plt.plot(x_extension, y_extension)
#                 # plt.xlim([199e12, 201e12])
#                 plt.show()
#                 pass
#             elif m[i, j] < 0:
#                 y_ext = extend_down(m[i, j], b[i, j])
#             else:
#                 pass
#                 # y_ext = y[-1, i, j]*jnp.ones(N)


# def extend_s_params(s_params, f, f_extended, alpha=1e11):
#     magnitude = jnp.abs(s_params)
#     phase = jnp.unwrap(jnp.angle(s_params), axis=0)
#     phase_slope, phase_intercept = line_of_best_fit_m(f, phase)
#     avg_phase = phase_slope[None, :, :]*f[:, None, None] + phase_intercept[None, :, :]
#     normalized_phase = phase - avg_phase
#     bandwidth = 0.5*jnp.abs(f[-1] - f[0])
#     magnitude_extended = extend(f, magnitude, f_extended[0], f_extended[-1], bandwidth/10)
#     normalized_phase_extended = extend(f, phase)
#     pass


#     pass

# def tukey_freq_window(freqs, fc, trans_width, alpha=None):
#     """
#     Create a Tukey-like taper in frequency domain.

#     freqs: array of frequency points (can be positive or two-sided)
#     fc: flat passband edge (Hz)
#     trans_width: width of transition region (Hz)
#     alpha: fraction of total width for cosine taper; if None, computed from trans_width
#     """
#     W = jnp.zeros_like(freqs, dtype=float)
#     f_abs = jnp.abs(freqs)  # symmetric in frequency

#     # Passband region
#     pass_region = f_abs <= fc
#     W = W.at[pass_region].set(1.0)

#     # Transition region
#     trans_region = (f_abs > fc) & (f_abs < fc + trans_width)
#     x = (f_abs[trans_region] - fc) / trans_width  # 0 → 1 over transition
#     W = W.at[trans_region].set(0.5 * (1 + jnp.cos(jnp.pi * x)))

#     # Stopband region stays 0
#     return W

# def expand_filter_to_mimo(A_f, B_f, C_f, D_f, num_ports):
#     """Creates a block-diagonal MIMO filter from a single SISO filter."""
#     A = jax.scipy.linalg.block_diag(*[A_f] * num_ports)
#     B = jnp.zeros((A.shape[0], num_ports))
#     C = jnp.zeros((num_ports, A.shape[0]))
#     D = jnp.eye(num_ports) * D_f  # diagonal D matrix

#     n = A_f.shape[0]  # order of filter
#     for i in range(num_ports):
#         B = B.at[i*n:(i+1)*n, i].set(B_f[:, 0])
#         C = C.at[i, i*n:(i+1)*n].set(C_f[0, :])

#     return A, B, C, D

# def cascade_state_space(A1, B1, C1, D1, A2, B2, C2, D2):
#     n1 = A1.shape[0]
#     n2 = A2.shape[0]

#     A = jnp.block([
#         [A1,                   jnp.zeros((n1, n2))],
#         [B2 @ C1,              A2]
#     ])

#     B = jnp.vstack([
#         B1,
#         B2 @ D1
#     ])

#     C = jnp.hstack([
#         D2 @ C1,
#         C2
#     ])

#     D = D2 @ D1

#     return A, B, C, D


class Signal:  ## TODO: Make an actual base class
    ...


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

    @classmethod
    def _create_port_lookup_table(cls):
        """Build internal lookup tables from the component's declared ports.

        Bidirectional ports are included in both the input and output
        lookup tables because they can participate on either side
        depending on the simulation mode and netlist orientation.
        """
        cls._port_lookup_table = {p.name: p for p in cls.ports}
        # cls._input_port_lookup_table = {p.name: p  for p in cls.ports if p.directionality=="input"}
        # cls._output_port_lookup_table = {p.name: p for p in cls.ports if p.directionality=="output"}
        cls._input_port_lookup_table = {
            p.name: p
            for p in cls.ports
            if p.directionality == "input" or p.directionality == "bidirectional"
        }
        cls._output_port_lookup_table = {
            p.name: p
            for p in cls.ports
            if p.directionality == "output" or p.directionality == "bidirectional"
        }

    def __init__(
        self,
        simulation_mode: SimulationMode,
        **kwargs,
    ):
        raise ValueError("Component is a base class")

    # simulation_parameters={}
    # Used especially in time-domain simulations

    # electrical_ports = []
    # logic_ports = []
    # optical_ports = []


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
    supplies a dictionary of input signals keyed by port name, and the
    method should return a dictionary of output signals keyed by output
    port name.
    """

    # IDK the best name for this method! Maybe run, but that is confusing
    def block_mode_response(
        self,
        input_signals: ArrayLike,
        simulation_parameters: BlockModeSimulationParameters,
    ):
        """Compute output signals for one full Block mode time block.

        Parameters
        ----------
        input_signals:
            Mapping from input port name to a block signal object. Optical ports
            receive `BlockModeOpticalSignal`; electrical ports receive
            `BlockModeElectricalSignal`.
        simulation_parameters:
            Shared Block mode parameters defining the time grid, wavelengths,
            and modes.

        Returns
        -------
        dict
            Mapping from output port name to block signal object.
        """
        raise NotImplementedError

    def _block_mode_response(self, input_signals, simulation_parameters):
        for port_name, port in self._input_port_lookup_table.items():
            input_signals.setdefault(
                port_name, self._default_input_signal(simulation_parameters, port.type)
            )
        outputs = self.block_mode_response(input_signals, simulation_parameters)

        baseband_wls = simulation_parameters.optical_baseband_wavelengths
        time_steps = (
            jnp.arange(simulation_parameters.num_time_steps) * simulation_parameters.dt
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
                * jnp.exp(-1j * 2 * jnp.pi * f_diff[:, None] / f_s * time_step)
            )
            outputs[port_name] = signal.replace(
                amplitude=new_amplitude, wavelength=baseband_wls
            )

        return outputs, (time_step + 1, output_state)


# TODO: Get rid of this
class SParameterComponent(Component):
    """Mixin for components described by wavelength-dependent S-parameters.

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


# # TODO: Get rid of this
# class OpticalSParameterComponent(SParameterComponent):
#     # def __init__(self, **settings):
#     #     super().__init__(**settings)

#     def s_parameters(
#         self,
#         wl: ArrayLike,
#         # **kwargs
#     ):
#         """
#         Returns an S-parameter matrix for the optical ports in the system
#         """
#         raise NotImplementedError(
#             f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
#         )
