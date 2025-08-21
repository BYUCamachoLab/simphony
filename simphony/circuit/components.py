###### Necessary ######
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation.block_mode import BlockModeSimulationParameters  # Only imported for type checking
    from simulation.sample_mode import SampleModeSimulationParameters 
#### Include First ####

import inspect
from typing import Tuple
from dataclasses import replace


# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
# import gravis as gv
import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel
from simphony.time_domain.pole_residue_model import IIRModelBaseband
# from simphony.simulation.simulation import SimulationParameters

from simphony.utils import dict_to_matrix
# from simphony.simulation import SampleModeSimulationParameters
from copy import deepcopy
from functools import partial
from simphony.signals import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal

from scipy.constants import speed_of_light
from scipy.signal import butter, tf2ss, StateSpace, firwin, freqz, group_delay, cheby1, bessel
from scipy.signal.windows import tukey
from control import balred, ss
import matplotlib.pyplot as plt
from simphony.time_domain.vector_fitting.z_domain import vector_fitting_z_optimize_order, pole_residue_response, state_space_z

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
from simphony.signals import BlockModeElectricalSignal, BlockModeLogicSignal, BlockModeOpticalSignal, SteadyStateOpticalSignal

import jax
import jax.numpy as jnp

from simphony.utils import dict_to_matrix
from jax.scipy.special import i0
from scipy.special import lambertw

from scipy.signal.windows import kaiser_bessel_derived

def line_of_best_fit_m(x, y):
    x_mean = jnp.mean(x[:, None, None], axis=0)
    y_mean = jnp.mean(y, axis=0)
    cov = jnp.mean((x[:, None, None]-x_mean)*(y - y_mean), axis=0)
    var = jnp.mean((x[:, None, None]-x_mean)**2, axis=0)
    slope = cov/var
    intercept = y_mean - slope * x_mean
    return slope, intercept

def _extension_up(m, b, x, y_initial, y_final):
    k = m*(y_final - y_initial)
    
    x_ext = x - x[0]
    y_ext = y_final + (y_initial - y_final)*jnp.exp(-k*x)
    return y_ext

def extend_down(m, b, y_f=0.0, N=500):
    pass

def extend(x, y, x_min, x_max, alpha=1e11):
    dy = y[-1] - y[-2]
    dx = x[-1] - x[-2]
    m = dy/dx
    b = y[-1] - m*x[-1]
    
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            if m[i, j] > 0:
                y_initial = y[-1, i, j]
                p = 1 - jnp.exp(-alpha*m[i, j])
                y_final = y_initial + p*(1-y_initial)
                x_extension = jnp.arange(x[-1]+dx, x_max, dx)
                y_extension = _extension_up(m[i, j], b[i, j], x_extension, y_initial, y_final)

                x_extended = jnp.concatenate([x, x_extension])
                y_extended = jnp.concatenate([y[:, i, j], y_extension])
                # plt.plot(x_extended, y_extended)
                # plt.plot(x, y[:, i, j])
                plt.plot(x_extension, y_extension)
                # plt.xlim([199e12, 201e12])
                plt.show()
                pass
            elif m[i, j] < 0:
                y_ext = extend_down(m[i, j], b[i, j])
            else:
                pass
                # y_ext = y[-1, i, j]*jnp.ones(N)




def extend_s_params(s_params, f, f_extended, alpha=1e11):
    magnitude = jnp.abs(s_params)
    phase = jnp.unwrap(jnp.angle(s_params), axis=0)
    phase_slope, phase_intercept = line_of_best_fit_m(f, phase)
    avg_phase = phase_slope[None, :, :]*f[:, None, None] + phase_intercept[None, :, :]
    normalized_phase = phase - avg_phase
    bandwidth = 0.5*jnp.abs(f[-1] - f[0])
    magnitude_extended = extend(f, magnitude, f_extended[0], f_extended[-1], bandwidth/10)
    normalized_phase_extended = extend(f, phase)
    pass

    
    pass

def tukey_freq_window(freqs, fc, trans_width, alpha=None):
    """
    Create a Tukey-like taper in frequency domain.

    freqs: array of frequency points (can be positive or two-sided)
    fc: flat passband edge (Hz)
    trans_width: width of transition region (Hz)
    alpha: fraction of total width for cosine taper; if None, computed from trans_width
    """
    W = jnp.zeros_like(freqs, dtype=float)
    f_abs = jnp.abs(freqs)  # symmetric in frequency

    # Passband region
    pass_region = f_abs <= fc
    W = W.at[pass_region].set(1.0)

    # Transition region
    trans_region = (f_abs > fc) & (f_abs < fc + trans_width)
    x = (f_abs[trans_region] - fc) / trans_width  # 0 → 1 over transition
    W = W.at[trans_region].set(0.5 * (1 + jnp.cos(jnp.pi * x)))

    # Stopband region stays 0
    return W

def expand_filter_to_mimo(A_f, B_f, C_f, D_f, num_ports):
    """Creates a block-diagonal MIMO filter from a single SISO filter."""
    A = jax.scipy.linalg.block_diag(*[A_f] * num_ports)
    B = jnp.zeros((A.shape[0], num_ports))
    C = jnp.zeros((num_ports, A.shape[0]))
    D = jnp.eye(num_ports) * D_f  # diagonal D matrix

    n = A_f.shape[0]  # order of filter
    for i in range(num_ports):
        B = B.at[i*n:(i+1)*n, i].set(B_f[:, 0])
        C = C.at[i, i*n:(i+1)*n].set(C_f[0, :])

    return A, B, C, D

# def cascade_state_space(A1, B1, C1, D1, A2, B2, C2, D2):
#     n1 = A1.shape[0]
#     n2 = A2.shape[0]

#     A = jnp.block([
#         [A1,                  jnp.zeros((n1, n2))],
#         [B2 @ C1,             A2]
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

def cascade_state_space(A1, B1, C1, D1, A2, B2, C2, D2):
    n1 = A1.shape[0]
    n2 = A2.shape[0]

    A = jnp.block([
        [A1,                   jnp.zeros((n1, n2))],
        [B2 @ C1,              A2]
    ])

    B = jnp.vstack([
        B1,
        B2 @ D1
    ])

    C = jnp.hstack([
        D2 @ C1,
        C2
    ])

    D = D2 @ D1

    return A, B, C, D


class Signal: ## TODO: Make an actual base class
    ...

# class SimulationParameters:
#     def __init__(
#         self,
#         sampling_period=None,
#         sampling_rate=None,
#         num_time_steps=None,
#     ):
#         self.sampling_period = sampling_period
#         self.sampling_rate = sampling_rate
#         self.num_time_steps = num_time_steps

class Component:
    # simulation_parameters={}
    delay_compensation = 0 # Used especially in time-domain simulations
    
    electrical_ports = []
    logic_ports = []
    optical_ports = []
    
    # def set_simulation_parameters(
    #     self,
    #     # optical_wls=[1.55e-6],
    #     # electrical_wls=[0],
    #     sampling_period=1e-15,
    #     num_time_steps=10000,
    # ):
    #     # self.simulation_parameters['sampling_period'] = sampling_period
    #     # self.simulation_parameters['num_time_steps'] = num_time_steps
    #     # self.simulation_parameters['delay_compensation'] = delay_compensation
    #     self.simulation_parameters = SimulationParameters(
    #         sampling_period=sampling_period,
    #         sampling_rate=1/sampling_period,
    #         num_time_steps=int(num_time_steps),
    #     )
        # self.simulation_parameters['optical_wls'] = optical_wls
        # self.simulation_parameters['electrical_wls'] = electrical_wls
    # def __init__(self, **settings):
    #     self.settings = settings
    #     for key, value in settings.items():
    #         setattr(self, key, value)

class SteadyStateComponent(Component):
    """ 
    """
    # def __init__(self, **settings):
    #     super().__init__(**settings)

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
    def __init__(
        self
        # , optical_ports=None, electrical_ports=None, logic_ports=None
    ) -> None:
        ...
        # super().__init__(optical_ports, electrical_ports, logic_ports)

    # IDK the best name for this method! Maybe run, but that is confusing
    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters: BlockModeSimulationParameters):
        """Compute the system response."""
        raise NotImplementedError


class SampleModeComponent(Component):
    # def _sample_mode_restart(
    #     self,
    #     # optical_wls, 
    #     # electrical_wls,
    #     # sampling_period,
    #     # num_time_steps,
    #     max_delay_compensation,
    # ):
        # # self.sample_mode_simulation_parameters(
        # #     # optical_wls, 
        # #     # electrical_wls,
        # #     sampling_period,
        # #     num_time_steps, 
        # # )
        # T = max_delay_compensation - self.delay_compensation + 1
        # num_optical_wls = len(optical_wls)
        # num_electrical_wls = len(electrical_wls)
        # buffered_outputs = {}
        # for electrical_port in self.electrical_ports:
        #     buffered_outputs[electrical_port] = BlockModeElectricalSignal(
        #         amplitude = jnp.zeros((T, num_electrical_wls), dtype=complex),
        #         wl = electrical_wls,
        #     )
        # for logic_port in self.logic_ports:
        #     buffered_outputs[logic_port] = BlockModeLogicSignal(
        #         value = jnp.zeros((T, ), dtype=int),
        #     )
        # for optical_port in self.optical_ports:
        #     buffered_outputs[optical_port] = BlockModeOpticalSignal(
        #         amplitude = jnp.zeros((T, num_optical_wls), dtype=complex),
        #         wl = optical_wls,
        #         # Use default polarization
        #     )
        
        # self._initial_buffered_outputs = buffered_outputs

    def sample_mode_initial_state(self, simulation_parameters: SampleModeSimulationParameters):
        """
        May be overwritten by user.
        Returns the initial the state of the system.
        Called by the sample mode simulator after `set_sample_mode_simulation_parameters`
        """
        return 0

    def _sample_mode_initial_state(self, max_delay_compensation, simulation_parameters: SampleModeSimulationParameters):
        optical_wls = simulation_parameters.optical_baseband_wavelengths
        electrical_wls = simulation_parameters.electrical_baseband_wavelengths
        num_optical_modes = simulation_parameters.num_optical_modes
        T = max_delay_compensation - self.delay_compensation + 1
        self._sample_mode_buffer_length = T
        num_optical_wls = len(optical_wls)
        num_electrical_wls = len(electrical_wls)
        buffered_outputs = {}
        for electrical_port in self.electrical_ports:
            buffered_outputs[electrical_port] = SampleModeElectricalSignal(
                amplitude = jnp.zeros((T, num_electrical_wls), dtype=complex),
                wavelength = jnp.tile(electrical_wls, (T, num_electrical_wls)),
            )
        for logic_port in self.logic_ports:
            buffered_outputs[logic_port] = SampleModeLogicSignal(
                value = jnp.zeros((T, ), dtype=int),
            )
        for optical_port in self.optical_ports:
            buffered_outputs[optical_port] = SampleModeOpticalSignal(
                amplitude = jnp.zeros((T, num_optical_wls, num_optical_modes), dtype=complex),
                wavelength = jnp.tile(optical_wls, (T, num_optical_wls)),
                # Use default polarization
            )
        
        _initial_state = (0, buffered_outputs, self.sample_mode_initial_state(simulation_parameters=simulation_parameters))
        return _initial_state
    
    # def set_sampling_period(self, sampling_period):
    #     self.sampling_period = sampling_period
    #     self.sampling_rate = 1 / sampling_period
    
    # def set_num_time_steps(self, num_time_steps):
    #     self.num_time_steps = num_time_steps

    
    # def prestep(self, inputs):
    #     """
    #     Should be used in sample-mode simulations in which the 
    #     inputs and outputs are dynamic, i.e. simulations with 
    #     Nonlinear components. 

    #     This function should ensure that the input optical signals 
    #     each have the same dimensions (same wavelengths), and 
    #     similarly, the input electrical signals if there are any.
            
    #     If any objects or variables must be instantiated for use
    #     in the `step()` function as a result of their addition here,
    #     those objects must be instantiated in `prestep()` 
        
    #     This ensures that the step function can be a pure function,
    #     free of side-effects.
    #     """
    #     complete_sample_mode_inputs(inputs)

    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters: SampleModeSimulationParameters) -> Tuple[jax.Array, dict[str, Signal]]:
        """Compute the next state of the system."""
        raise NotImplementedError
    
    # @partial(jax.jit, static_argnums=(0,))
    def _sample_mode_step(self, inputs: dict, state: jax.Array, simulation_parameters: SampleModeSimulationParameters):
        time_step = state[0]
        # prng_key = state[1]
        buffered_outputs = state[1]
        internal_state = state[2]

        # We can ensure that each step function is called with a unique key
        # new_prng_key, subkey = jax.random.split(prng_key)
        # new_simulation_parameters = replace(simulation_parameters,prng_key=subkey)
        buffer_index = time_step % self._sample_mode_buffer_length
        pass
        outputs, output_state = self.sample_mode_step(inputs, internal_state, simulation_parameters)
        for port in buffered_outputs.keys():
            amplitude = outputs[port].amplitude
            amplitude_block = buffered_outputs[port].amplitude.at[buffer_index, :, :].set(amplitude)
            wavelength = outputs[port].wavelength
            wavelength_block = buffered_outputs[port].wavelength.at[buffer_index, :].set(wavelength)
            buffered_outputs[port] = buffered_outputs[port].replace(amplitude=amplitude_block, wavelength=wavelength_block)

        delayed_outputs = jax.tree_util.tree_map(lambda leaf: leaf[(buffer_index+1)%self._sample_mode_buffer_length], buffered_outputs)

        return delayed_outputs, (time_step+1, buffered_outputs, output_state)

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


def _optical_s_parameter(sax_model: SaxModel):
    class SParameterSax(OpticalSParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent):
        optical_ports = list(sax.get_ports(sax_model))
        _num_ports = len(optical_ports)
        
        def __init__(
            self, 
            spectral_range=(1.5e-6,1.6e-6),
            method = 'optimal_order',
            **sax_settings
        ):
            # super().__init__(**settings)
            self.settings = sax_settings
            self.spectral_range = spectral_range
            self.port_order = {name: idx for idx, name in enumerate(self.optical_ports)}

            if method == 'optimal_order':
                self.sample_mode_initial_state = self.sample_mode_initial_state_optimal_order
                self.sample_mode_step = self.sample_mode_step_optimal_order
            elif method == 'ss_fir_filt':
                self.sample_mode_initial_state = self.sample_mode_initial_state_ss_fir_filt
                self.sample_mode_step = self.sample_mode_step_ss_fir_filt
                self.fir_length = 11
                self.delay_compensation = int((self.fir_length - 1)/2)
            elif method == 'h_fir_filt':
                self.sample_mode_initial_state = self.sample_mode_initial_state_h_fir_filt
                self.sample_mode_step =  self.sample_mode_step_h_fir_filt
            elif method == 'ss_windowed':
                self.sample_mode_initial_state = self.sample_mode_initial_state_ss_windowed
                self.sample_mode_step = self.sample_mode_step_ss_windowed

        def sample_mode_initial_state_optimal_order(
            self,
            simulation_parameters,
        ):
            N = 1000
            f_min = speed_of_light / self.spectral_range[1]
            f_max = speed_of_light / self.spectral_range[0]
            
            # f = jnp.linspace(f_min, f_max, N)
            f_c = 0.5*(f_max + f_min)
            f_s = 1 / simulation_parameters.sampling_period
            beta = f_s / (f_max - f_min)
            f_b_padded = jnp.linspace(-0.5*f_s, 0.5*f_s, int(beta*N))
            f_padded = f_b_padded + f_c
            start = int(beta * N - N) // 2
            f_b = f_b_padded[start:start + N]
            f = f_b + f_c
            s_params = dict_to_matrix(self.s_parameters(wl=speed_of_light/f))
            s_params_extended = extend_s_params(s_params, f, f_padded)
            
            phase = jnp.unwrap(jnp.angle(s_params), axis=0)

            plt.plot(f, s_params[:, 0, 1])
            plt.show()

            ###
            # Find the average phase velocity
            ###
            f_mean = jnp.mean(f[:, None, None], axis=0)
            phase_mean = jnp.mean(phase, axis=0)
            cov = jnp.mean((f[:, None, None]-f_mean)*(phase - phase_mean), axis=0)
            var = jnp.mean((f[:, None, None]-f_mean)**2, axis=0)
            slope = cov/var
            intercept = phase_mean - slope * f_mean
            
            phase_padded = slope[None, :, :]*f_padded[:, None, None] + intercept[None, :, :]
            
            
            plt.plot(f_padded, phase_padded[:, 0, 1])
            plt.plot(f, phase[:, 0, 1])
            plt.show()

            s_params_padded = jnp.exp(1j*phase_padded)
            
            start = int(beta * N - N) // 2
            s_params_padded = s_params_padded.at[start:start + N].set(s_params)
            bandwidth = 0.5*(f_max - f_min)
            W = tukey_freq_window(f_padded - f_c, bandwidth, 50e12)
            
            W = jnp.tile(W[:, None, None], (1, 2, 2))
            s_params_padded *= W
            s_params_padded *= jnp.mean(jnp.abs(s_params), axis=0)
            zero_region_step = 1500
            buffer = 3500
            s_params_reduced = jnp.concatenate((s_params_padded[0:start-buffer:zero_region_step] , s_params_padded[start-buffer:start + N+buffer], s_params_padded[start + N+buffer::zero_region_step]), axis=0)
            f_reduced = jnp.concatenate((f_padded[0:start-buffer:zero_region_step] , f_padded[start-buffer:start + N+buffer], f_padded[start + N+buffer::zero_region_step]), axis=0)
            # zero_region = 20
            # s_params_reduced = jnp.concatenate((s_params_padded[:zero_region] , s_params_padded[start:start + N], s_params_padded[-zero_region:]), axis=0)
            # f_reduced = jnp.concatenate((f_padded[:zero_region] , f_padded[start:start + N], f_padded[-zero_region:]), axis=0)
            poles, residues, feedthrough, _ = vector_fitting_z_optimize_order(2, 80, s_params_reduced, f_reduced, f_c, f_s)
            A, B, C, D = state_space_z(poles, residues, feedthrough)
            self.state_space_model = (A, B, C, D)
            
            H = pole_residue_response(f, f_c, f_s, poles, residues, feedthrough)
            H_full = pole_residue_response(jnp.linspace(-f_s/2, f_s/2, 1000)+f_c, f_c, f_s, poles, residues, feedthrough)
            print(f"NUMBER OF POLES: {len(poles)}")
            plt.plot(f, jnp.abs(H[:, 0, 1])**2)
            plt.show()
            plt.plot(jnp.linspace(-f_s/2, f_s/2, 1000), jnp.abs(H_full[:, 0, 1])**2)
            plt.show()
            return jnp.zeros((A.shape[0],), dtype=complex)
        
        def sample_mode_step_optimal_order(
            self,
            inputs: dict,
            state: jax.Array,
            simulation_parameters,
        ):
            x = state
            A, B, C, D = self.state_space_model
            
            u = jnp.zeros((len(self.optical_ports),),dtype=complex)
            TE_MODE = 0
            for port, signal in inputs.items():
                port_idx = self.port_order[port]
                wavelength = inputs[port].wavelength[0]
                u = u.at[port_idx].set(signal.amplitude[0, TE_MODE])
            
            new_x = A@x + B@u
            y = C@x + D@u

            outputs = {}
            for port in self.optical_ports:
                A_t = y[self.port_order[port]]
                outputs[port] = SampleModeOpticalSignal(
                    ### TODO: FIX THIS FOR MULTIPLE FREQUENCIES
                    ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
                    amplitude = A_t.reshape((1, 1)),
                    wavelength = jnp.array([inputs[port].wavelength[0]])
                )

            return outputs, new_x 

        def sample_mode_initial_state_ss_windowed(
            self, 
            simulation_parameters,
        ):
            N = 1000
            f_min = speed_of_light / self.spectral_range[1]
            f_max = speed_of_light / self.spectral_range[0]
            
            # f = jnp.linspace(f_min, f_max, N)
            f_c = 0.5*(f_max + f_min)
            f_s = 1 / simulation_parameters.sampling_period
            beta = f_s / (f_max - f_min)
            f_b_padded = jnp.linspace(-0.5*f_s, 0.5*f_s, int(beta*N))
            f_padded = f_b_padded + f_c
            start = int(beta * N - N) // 2
            f_b = f_b_padded[start:start + N]
            f = f_b + f_c
            s_params = dict_to_matrix(self.s_parameters(wl=speed_of_light/f))
            
            s_params_padded = jnp.zeros((f_padded.shape[0], s_params.shape[1], s_params.shape[2]), dtype=complex)
            
            start = int(beta * N - N) // 2
            s_params_padded = s_params_padded.at[start:start + N].set(s_params)
            bandwidth = 0.5*(f_max - f_min)
            # W = tukey_freq_window(f_padded - f_c, bandwidth, 50e12)
            # W = kbd_freq_window(f_padded - f_c, 0.5*bandwidth, 1e12)
            W = kaiser_bessel_derived(len(s_params), 31.83)
            W = jnp.tile(W[:, None, None], (1, s_params.shape[1], s_params.shape[2]))
            plt.plot(f_padded, s_params_padded[:, 0 , 1])
            plt.plot(f, W[:, 0 , 1])
            plt.xlim([1.7e14, 2.2e14])
            plt.show()
            # s_params_padded = W
            
            # s_params_padded *= jnp.mean(jnp.abs(s_params), axis=0)
            zero_region_step = 1000
            s_params_reduced = jnp.concatenate((s_params_padded[0:start:zero_region_step], W*s_params_padded[start:start + N], s_params_padded[start + N::zero_region_step]), axis=0)
            f_reduced = jnp.concatenate((f_padded[0:start:zero_region_step], f_padded[start:start + N], f_padded[start + N::zero_region_step]), axis=0)
            # zero_region = 20
            # s_params_reduced = jnp.concatenate((s_params_padded[:zero_region] , s_params_padded[start:start + N], s_params_padded[-zero_region:]), axis=0)
            # f_reduced = jnp.concatenate((f_padded[:zero_region] , f_padded[start:start + N], f_padded[-zero_region:]), axis=0)
            poles, residues, feedthrough, _ = vector_fitting_z_optimize_order(2, 80, s_params_reduced, f_reduced, f_c, f_s)
            A, B, C, D = state_space_z(poles, residues, feedthrough)
            self.state_space_model = (A, B, C, D)
            
            H = pole_residue_response(f, f_c, f_s, poles, residues, feedthrough)
            H_full = pole_residue_response(jnp.linspace(-f_s/2, f_s/2, 1000)+f_c, f_c, f_s, poles, residues, feedthrough)
            print(f"NUMBER OF POLES: {len(poles)}")
            plt.plot(f, jnp.abs(H[:, 0, 1])**2)
            plt.show()
            plt.plot(jnp.linspace(-f_s/2, f_s/2, 1000), jnp.abs(H_full[:, 0, 1])**2)
            plt.show()
            return jnp.zeros((A.shape[0],), dtype=complex)
        
        def sample_mode_step_ss_windowed(
            self,
            inputs: dict,
            state: jax.Array,
            simulation_parameters,
        ):
            x = state
            A, B, C, D = self.state_space_model
            
            u = jnp.zeros((len(self.optical_ports),),dtype=complex)
            TE_MODE = 0
            for port, signal in inputs.items():
                port_idx = self.port_order[port]
                wavelength = inputs[port].wavelength[0]
                u = u.at[port_idx].set(signal.amplitude[0, TE_MODE])
            
            new_x = A@x + B@u
            y = C@x + D@u

            outputs = {}
            for port in self.optical_ports:
                A_t = y[self.port_order[port]]
                outputs[port] = SampleModeOpticalSignal(
                    ### TODO: FIX THIS FOR MULTIPLE FREQUENCIES
                    ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
                    amplitude = A_t.reshape((1, 1)),
                    wavelength = jnp.array([inputs[port].wavelength[0]])
                )

            return outputs, new_x 

        # def sample_mode_initial_state_ss_windowed(
        #     self, 
        #     simulation_parameters,
        # ):
        #     self.state_space_models = []

        #     f_min = speed_of_light / self.spectral_range[1]
        #     f_max = speed_of_light / self.spectral_range[0]
        #     center_freq = 0.5*(f_max + f_min)
        #     center_wl = speed_of_light / center_freq
        #     beta = 100.0
        #     N = 1000
        #     fs = beta*(f_max-f_min)
        #     baseband_frequency = jnp.linspace(f_min-center_freq, f_max-center_freq, N)
        #     baseband_frequency_padded = jnp.linspace(-0.5*fs, 0.5*fs, int(beta*N))
        #     # wvl = jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
        #     s_params = dict_to_matrix(self.s_parameters(wl=speed_of_light/(baseband_frequency + center_freq)))
        #     s_params_padded = jnp.zeros((int(beta*N), s_params.shape[1], s_params.shape[2]), dtype=s_params.dtype)
        #     start = int(beta * N - N) // 2
        #     s_params_padded = s_params_padded.at[start:start + N].set(s_params)

        #     # baseband_frequency = speed_of_light/wvl - center_freq
        #     bandwidth = 0.5*(f_max - f_min)
        #     W = tukey_freq_window(baseband_frequency, bandwidth, 1e12)
            
        #     W = jnp.tile(W[:, None, None], (1, 2, 2))
        #     plt.plot(baseband_frequency_padded, jnp.abs((s_params_padded)[:, 0, 1])**2)
        #     plt.plot(baseband_frequency, jnp.abs((s_params)[:, 0, 1])**2)
        #     plt.show()
        #     pass
        #     # N = s_params.shape[0]
        #     # beta = 1
            

        #     plt.plot(jnp.abs(s_params_padded[:, 0, 1])**2)
        #     plt.show()

        #     zero_region = 10
        #     s_params_reduced = jnp.concatenate((s_params_padded[:zero_region] , s_params_padded[start:start + N], s_params_padded[-zero_region:]), axis=0)
        #     baseband_frequency_reduced = jnp.concatenate((baseband_frequency_padded[:zero_region] , baseband_frequency_padded[start:start + N], baseband_frequency_padded[-zero_region:]), axis=0)
            
        #     # s_params_reduced = s_params_padded[start:start + N]
        #     # baseband_frequency_reduced = baseband_frequency_padded[start:start + N]




        #     plt.plot(baseband_frequency_reduced, s_params_reduced[:, 0, 1])
        #     plt.show()


        #     z_domain_model = IIRModelBaseband(
        #             wvl_microns=jnp.flip(1e6*speed_of_light/(baseband_frequency_reduced+center_freq)),
        #             center_wvl=1e6*center_wl, 
        #             s_params=jnp.flip(s_params_reduced), 
        #             sampling_period=1/fs, 
        #             order=30
        #         )
            
        #     # baseband_frequency = jnp.linspace(-0.5*fs, 0.5*fs, 1500)
        #     H = z_domain_model.baseband_transfer_function(baseband_frequency_padded)
        #     plt.plot(baseband_frequency_padded, jnp.abs(H[:, 0, 1])**2)
        #     plt.ylim([0.0, 2.0])
        #     plt.show()
            
        #     fs = 1/simulation_parameters.sampling_period
        #     baseband_frequency = jnp.linspace(-0.5*fs, 0.5*fs, 1500)
        #     H = z_domain_model.baseband_transfer_function(baseband_frequency)
            
            
        #     plt.plot(baseband_frequency, jnp.abs(H[:, 0, 1])**2)
        #     plt.show()
            
        #     # W = tukey_freq_window(baseband_frequency, f_max - center_freq, 1e14)
        #     # W = jnp.tile(W[:, None, None], (1, 2, 2))
        #     # plt.plot(baseband_frequency, jnp.abs((W*H)[:, 0, 1])**2)
        #     # plt.show()
        #     # z_domain_model = IIRModelBaseband(
        #     #         wvl_microns=1e6*(speed_of_light / baseband_frequency + center_wl),
        #     #         center_wvl=1e6*center_wl, 
        #     #         s_params=W*H, 
        #     #         sampling_period=simulation_parameters.sampling_period, 
        #     #         order=40
        #     #     )

            
        #     # z_domain_model.plot()
        #     # plt.show()
        #     self.state_space_model = z_domain_model.generate_sys_discrete()

        #     return jnp.zeros((self.state_space_model.A.shape[0],), dtype=complex)
        
        # def sample_mode_step_ss_windowed(
        #     self,
        #     inputs: dict,
        #     state: jax.Array,
        #     simulation_parameters,
        # ):
        #     x = state
        #     A = self.state_space_model.A
        #     B = self.state_space_model.B
        #     C = self.state_space_model.C
        #     D = self.state_space_model.D
            
        #     u = jnp.zeros((len(self.optical_ports),),dtype=complex)
        #     TE_MODE = 0
        #     for port, signal in inputs.items():
        #         port_idx = self.port_order[port]
        #         wavelength = inputs[port].wavelength[0]
        #         u = u.at[port_idx].set(signal.amplitude[0, TE_MODE])
            
        #     new_x = A@x + B@u
        #     y = C@x + D@u

        #     outputs = {}
        #     for port in self.optical_ports:
        #         A_t = y[self.port_order[port]]
        #         outputs[port] = SampleModeOpticalSignal(
        #             ### TODO: FIX THIS FOR MULTIPLE FREQUENCIES
        #             ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
        #             amplitude = A_t.reshape((1, 1)),
        #             wavelength = jnp.array([inputs[port].wavelength[0]])
        #         )

        #     return outputs, new_x
            

        # def sample_mode_initial_state_fir(
        #     self, 
        #     simulation_parameters,
        # ):
        #     center_wl = 0.5*(self.spectral_range[1] - self.spectral_range[0])
        #     wvl = jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
        #     s_params = self.s_parameters(wl=1e6*wvl)
            
        #     z_domain_model = IIRModelBaseband(
        #             wvl_microns=1e6*wvl,
        #             center_wvl=1e6*center_wl, 
        #             s_params=dict_to_matrix(s_params), 
        #             sampling_period=simulation_parameters.sampling_period, 
        #             order=50
        #         )
            
        #     impulse_response = z_domain_model.discrete_time_impulse_response()
        #     pass
            
        
        def sample_mode_initial_state_h_fir_filt(
            self, 
            simulation_parameters,
        ):
            self.state_space_models = []
            center_wl = 0.5*(self.spectral_range[1] + self.spectral_range[0])
            wvl = jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
            s_params = self.s_parameters(wl=wvl)
            
            z_domain_model = IIRModelBaseband(
                    wvl_microns=1e6*wvl,
                    center_wvl=1e6*center_wl, 
                    s_params=dict_to_matrix(s_params), 
                    sampling_period=simulation_parameters.sampling_period, 
                    order=20
                )
            
            self.impulse_response = z_domain_model.discrete_time_impulse_response()
            pass
            
        
        def sample_mode_step_h_fir_filt(
            self,
            inputs: dict,
            state: jax.Array,
            simulation_parameters,
        ):
            # TODO: Use a Fourier-Based Approach
            raise NotImplementedError("FIR Filters for S-parameter Elements Not Implemented")

        def sample_mode_initial_state_ss_fir_filt(self, simulation_parameters):
            self.state_space_models = []
            center_wl = 0.5*(self.spectral_range[1] + self.spectral_range[0])
            wvl = jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
            s_params = self.s_parameters(wl=wvl)
            
            z_domain_model = IIRModelBaseband(
                    wvl_microns=1e6*wvl,
                    center_wvl=1e6*center_wl, 
                    s_params=dict_to_matrix(s_params), 
                    sampling_period=simulation_parameters.sampling_period, 
                    order=25
                )
            state_space_model = z_domain_model.generate_sys_discrete()
            
            # A, B, C, D = state_space_model.A, state_space_model.B, state_space_model.C, state_space_model.D
            # epsilon = 1e-6
            # U, S, Vh = jnp.linalg.svd(C, full_matrices=False)
            # S_thresh = S * (S > epsilon)
            # C_filtered = (U * S_thresh) @ Vh
            # state_space_model = StateSpace(A, B, C_filtered, D)

            self.state_space_models.append(state_space_model)

            bandwidth = speed_of_light * (1/self.spectral_range[0] - 1/self.spectral_range[1] )
            fs = 1 / simulation_parameters.sampling_period
            fir_cutoff = 0.5*bandwidth / (0.5*fs)
            
            h_np = firwin(self.fir_length, fir_cutoff, window='hamming')
            # h_np = firwin(fir_length, 0.0006, window='boxcar')
            self.fir_impulse_response = jnp.array(h_np)
            fir_buffer = jnp.zeros((len(self.optical_ports), self.fir_length), dtype=complex)
            
            
            time_step = 0
            return (time_step, fir_buffer, jnp.zeros((self.state_space_models[0].A.shape[0],), dtype=complex))

            #     # Convert pole-resiude model to state space model
                
            #     num_ports = len(self.optical_ports)

            #     # 1. Design IIR filter
            #     # b, a = butter(4, 0.0035)
            #     b, a = bessel(7, 0.008)
            #     # Compute group delay
            #     w, h = freqz(b, a)
            #     phase = jnp.unwrap(jnp.angle(h))
            #     freq = 1e15*w / jnp.pi
            #     group_delay = -jnp.gradient(phase)/jnp.gradient(freq)
            #     # print(group_delay)


            #     A1_siso, B1_siso, C1_siso, D1_siso = tf2ss(b, a)
            #     D1_siso = jnp.zeros_like(D1_siso)


            #     # FIR filter design using Hamming window
            #     # num_taps = 51           # Filter order + 1
            #     # cutoff = 0.005            # Normalized cutoff frequency (0.5 = Nyquist)
            #     # b = firwin(num_taps, cutoff, window='hamming')
            #     # a = [1.0] + [0.0] * (len(b) - 1)               # FIR filters have denominator = 1

            #     # Plot frequency response
            #     w, h = freqz(b, a)
            #     import matplotlib.pyplot as plt
            #     plt.plot(1e15*w / jnp.pi, 20 * jnp.log10(jnp.abs(h)))
            #     # plt.plot(1e15*w / jnp.pi, jnp.angle(h))
            #     plt.title("FIR Filter with Hamming Window")
            #     plt.xlabel("Normalized Frequency")
            #     plt.xlim([0, 20e12])
            #     plt.ylabel("Magnitude (dB)")
            #     plt.grid(True)
            #     plt.show()
                
            #     def H_state_space(A, B, C, D, omega):
            #         """Return frequency response H(jω) given state-space matrices."""
            #         s = 1j * omega
            #         I = jnp.eye(A.shape[0])
            #         H = C @ jnp.linalg.pinv(s * I - A) @ B + D
            #         return H
                
            #     # 2. Expand to MIMO filter
            #     A1, B1, C1, D1 = expand_filter_to_mimo(A1_siso, B1_siso, C1_siso, D1_siso, num_ports)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
                
            #     omega_vals = 2*jnp.pi*freq
            #     H_vals = jnp.array([H_state_space(A1, B1[:, 0:0+1], C1[0:0+1, :], D1[0, 0:0+1], w) for w in omega_vals])
            #     phase = jnp.unwrap(jnp.angle(H_vals))
            #     dphi_domega = jnp.gradient(phase[:, 1, 1], omega_vals)
            #     group_delay = -dphi_domega  # in seconds
            #     print(group_delay[0])

                
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # A1, B1, C1, D1 = cascade_state_space(A1, B1, C1, D1, A1, B1, C1, D1)
            #     # TODO: MAKE THE IIR FILTER Match the 

            #     # This is my code which generates the MIMO SYSTEM
            #     ss1 = iir_model.generate_sys_discrete()
            #     A2, B2, C2, D2 = ss1.A, ss1.B, ss1.C, ss1.D
                
            #     A, B, C, D = cascade_state_space(A1, B1, C1, D1, A2, B2, C2, D2)

            #     self.state_space_models.append(StateSpace(A, B, C, D))
            
            # return jnp.zeros((self.state_space_models[0].A.shape[0],), dtype=complex)
            
        # def prestep(self, inputs):
        #     """
        #     Should be used in sample-mode simulations in which the 
        #     inputs and outputs are dynamic, i.e. simulations with 
        #     Nonlinear components. 

        #     This function should ensure that the input optical signals 
        #     each have the same dimensions (same wavlengths), and 
        #     similarly, the input electrical signals if there are any.

        #     This ensures that the step function can be a pure function,
        #     free of side-effects.
        #     """
        #     complete_sample_mode_inputs(inputs)
        #     wls = inputs[self.optical_ports[0]].wl
        #     for wl in wls:
        #         # TODO: Make this more robust (floating point errors will hash differently)
        #         if not wl in self.state_space_models:
        #             # Create Pole-residue model
        #             wvl_microns = 1e6*jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
        #             center_wvl = wl

        #             # TODO: Make IIRModelBaseband take a sax.SDict
        #             s_params = sax_model(wvl_microns*1e6, **self.settings)
        #             iir_model = IIRModelBaseband(
        #                 wvl_microns=wvl_microns,
        #                 center_wvl=wl, 
        #                 s_params=s_params, 
        #                 sampling_period=self.sampling_period, 
        #                 order=50
        #             )
        #             # Convert pole-resiude model to state space model
        #             self.state_space_models[wl] = iir_model.to_sys_discrete()

        def sample_mode_step_ss_fir_filt(
            self,
            inputs: dict,
            state: jax.Array,
            simulation_parameters,
        ):
            time_step = state[0]
            fir_buffer = state[1]
            pass
            
            x = state[2]
            A = self.state_space_models[0].A
            B = self.state_space_models[0].B
            C = self.state_space_models[0].C
            D = self.state_space_models[0].D
            
            u = jnp.zeros((len(self.optical_ports),),dtype=complex)
            TE_MODE = 0
            for port, signal in inputs.items():
                port_idx = self.port_order[port]
                wavelength = inputs[port].wavelength[0]
                
                unfiltered_amplitude = signal.amplitude[0, TE_MODE]
                L = fir_buffer.shape[1]
                fir_buffer = fir_buffer.at[port_idx, time_step%L].set(unfiltered_amplitude)
                indices = (time_step - jnp.arange(L)) % L
                filtered_amplitude = jnp.sum(self.fir_impulse_response * fir_buffer[port_idx, indices])

                u = u.at[port_idx].set(filtered_amplitude)
            
            new_x = A@x + B@u
            y = C@x + D@u

            outputs = {}
            for port in self.optical_ports:
                A_t = y[self.port_order[port]]
                outputs[port] = SampleModeOpticalSignal(
                    ### TODO: FIX THIS FOR MULTIPLE FREQUENCIES
                    ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
                    amplitude = A_t.reshape((1, 1)),
                    wavelength = jnp.array([inputs[port].wavelength[0]])
                )

            return outputs, (time_step+1, fir_buffer, new_x)

        # @staticmethod
        # @jax.jit
        def s_parameters( 
            self,
            inputs: dict=None,
            wl: ArrayLike=1.55e-6,
        )->sax.SDict:
            # TODO: (MATTHEW! Don't do this one yet, I need to talk to Sequoia first)
            # Change the simphony models to be in units of meters not microns 
            return sax_model(wl*1e6, **self.settings)
        
        # @staticmethod
        # @jax.jit 
        def steady_state(self, inputs: dict):
            # Sadly, sax_model is not jit compatible
            # so instead we just jit what we can.
            # complete_steady_state_inputs(inputs)
            ports = sax.get_ports(sax_model)
            wl = inputs[ports[0]].wl
            s_params = dict_to_matrix(sax_model(wl*1e6, **self.settings))
            outputs = self._compute_outputs(s_params, wl, inputs)
            
            return outputs
        
        @staticmethod
        @jax.jit
        def _compute_outputs(s_params: ArrayLike, wls, inputs:dict)->dict:
            ports = sax.get_ports(sax_model)
            num_ports = len(ports)
            num_wls = wls.shape[0]
            input_matrix = jnp.zeros((num_wls, num_ports), dtype=complex)
            for i, port in enumerate(ports):
                input_matrix = input_matrix.at[:, i].set(inputs[port].field)
            
            output_matrix = jnp.zeros_like(input_matrix)
            for i, wl in enumerate(wls):
                _output = s_params[i,:,:] @ input_matrix[i, :]
                output_matrix = output_matrix.at[i, :].set(_output)

            outputs = {}
            for i, port in enumerate(ports):
                outputs[port] = SteadyStateOpticalSignal(
                                    field=output_matrix[:, i],
                                    wl=wls,
                                    polarization=inputs[port].polarization
                                )
            
            return outputs

    return SParameterSax