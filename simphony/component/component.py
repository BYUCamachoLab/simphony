###### Necessary ######
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.block_mode import BlockModeSimulationParameters  # Only imported for type checking
    from simulation.sample_mode import SampleModeSimulationParameters 
#### Include First ####

import inspect
from typing import Tuple
# from dataclasses import replace

# from simphony.circuit.port import Port


# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
# import gravis as gv
# import sax
from jax.typing import ArrayLike
# from sax.saxtypes import Model as SaxModel
# from simphony.time_domain.pole_residue_model import IIRModelBaseband
# from simphony.simulation.simulation import SimulationParameters

# from simphony.utils import dict_to_matrix
# from simphony.simulation import SampleModeSimulationParameters
# from copy import deepcopy
# from functools import partial
# from simphony.signals import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal

from scipy.constants import speed_of_light
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
from simphony.signal.block_mode import BlockModeElectricalSignal, BlockModeLogicSignal, BlockModeOpticalSignal
from simphony.signal.steady_state import SteadyStateOpticalSignal

import jax
import jax.numpy as jnp

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


class Signal: ## TODO: Make an actual base class
    ...

class Component:
    @classmethod
    def _create_port_lookup_table(cls):
        cls._port_lookup_table = {p.name: p for p in cls.ports}
    # simulation_parameters={}
     # Used especially in time-domain simulations
    
    # electrical_ports = []
    # logic_ports = []
    # optical_ports = []

class SteadyStateComponent(Component):
    """ 
    """
    delay_compensation = 0

    def steady_state(
        self, 
        inputs: dict
    ) -> dict:
        """
        Used when calculating steady state voltages for SParameterSimulation
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )


class BlockModeComponent(Component):
    # IDK the best name for this method! Maybe run, but that is confusing
    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters: BlockModeSimulationParameters):
        """Compute the system response."""
        raise NotImplementedError


class SampleModeComponent(Component):
    def sample_mode_initial_state(self, simulation_parameters: SampleModeSimulationParameters):
        """
        May be overwritten by user.
        Returns the initial the state of the system.
        Called by the sample mode simulator after `set_sample_mode_simulation_parameters`
        """
        return 0

    def _sample_mode_initial_state(self, simulation_parameters: SampleModeSimulationParameters):
        _initial_state = (0, self.sample_mode_initial_state(simulation_parameters=simulation_parameters))
        return _initial_state

    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters: SampleModeSimulationParameters) -> Tuple[jax.Array, dict[str, Signal]]:
        """Compute the next state of the system."""
        raise NotImplementedError
    
    # @partial(jax.jit, static_argnums=(0,))
    def _sample_mode_step(self, inputs: dict, state: jax.Array, simulation_parameters: SampleModeSimulationParameters):
        time_step = state[0]
        internal_state = state[1]
        
        f_s = 1/simulation_parameters.sampling_period

        outputs, output_state = self.sample_mode_step(inputs, internal_state, simulation_parameters)

        # Convert all inputs to the frequency channels in the simulator
        baseband_wls = simulation_parameters.optical_baseband_wavelengths
        for port, signal in outputs.items():
            amplitude = signal.amplitude
            wavelength = signal.wavelength
            dists = jnp.abs(baseband_wls[:, None] - wavelength[None, :])
            closest_idx = jnp.argmin(dists, axis=0)

            f_diff = speed_of_light / wavelength - speed_of_light / baseband_wls[closest_idx]

            new_amplitude = jnp.zeros((baseband_wls.shape[0], amplitude.shape[1]), dtype=complex)
            new_amplitude = new_amplitude.at[closest_idx].add(amplitude*jnp.exp(-1j*2*jnp.pi*f_diff[:, None]/f_s * time_step))
            outputs[port] = signal.replace(amplitude=new_amplitude, wavelength=baseband_wls)


        return outputs, (time_step+1, output_state)

class SParameterComponent(Component):
    """
    """
    def s_parameters(
        self,
        inputs: dict,
        wl: ArrayLike=1.55e-6,
    ):
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )

class OpticalSParameterComponent(SParameterComponent):
    # def __init__(self, **settings):
    #     super().__init__(**settings)

    def s_parameters(
        self, 
        wl: ArrayLike, 
        # **kwargs
    ):
        """
        Returns an S-parameter matrix for the optical ports in the system
        """
        raise NotImplementedError(
            f"{inspect.currentframe().f_code.co_name} method not defined for {self.__class__.__name__}"
        )


