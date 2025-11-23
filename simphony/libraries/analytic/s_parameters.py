from jax.typing import ArrayLike
import jax.numpy as jnp
import jax

from sax import get_ports
from sax.saxtypes import Model

import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

import matplotlib.pyplot as plt
from simphony.time_domain.vector_fitting.z_domain import optimize_order_vector_fitting_discrete, pole_residue_response_discrete, state_space_discrete
from simphony.signal.sample_mode import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal
from simphony.signal.steady_state import SteadyStateOpticalSignal

from scipy.constants import speed_of_light

from simphony.component.port import Port
from simphony.component.component import OpticalSParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent
from simphony.utils import dict_to_matrix

# optical_s_parameter is defined in simphony.circuit in order to maintain backwards compatability with
# sax models
# from simphony.circuit import _optical_s_parameter as optical_s_parameter


def optical_s_parameter(sax_model: SaxModel):
    optical_ports = list(sax.get_ports(sax_model()))
    class SParameterSax(OpticalSParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent):
        # optical_port_names = 
        # ports = {
        #     port_name: Port(name=port_name, type="optical", directionality="bidirectional") for port_name in optical_ports
        # }
        ports = [
            Port(name=port_name, type="optical", directionality="bidirectional") for port_name in optical_ports
        ]
        _num_ports = len(ports)
        
        def __init__(
            self, 
            spectral_range=(1.5e-6,1.6e-6),
            delay_compensation=0,
            max_error=1e-6,
            min_model_order=45,
            max_model_order=50,
            method = 'optimal_order',
            **sax_settings
        ):
            # super().__init__(**settings)
            self.max_error = max_error
            self.min_model_order = min_model_order
            self.max_model_order = max_model_order
            self.delay_compensation = delay_compensation
            self.settings = sax_settings
            self.spectral_range = spectral_range
            self.port_order = {name: idx for idx, name in enumerate(optical_ports)}

            if method == 'optimal_order':
                self.sample_mode_initial_state = self.sample_mode_initial_state_optimal_order
                self.sample_mode_step = self.sample_mode_step_optimal_order

        def sample_mode_initial_state_optimal_order(
            self,
            simulation_parameters,
        ):
            max_group_delay = 10e-12
            N = 1000
            f_min = speed_of_light / self.spectral_range[1]
            f_max = speed_of_light / self.spectral_range[0]
            
            f = jnp.linspace(f_min, f_max, N)
            f_c = 0.5*(f_max + f_min)
            f_s = 1 / simulation_parameters.sampling_period
            s_params = dict_to_matrix(self.s_parameters(wl=speed_of_light/f))
            
            M = 10000
            f_partial = jnp.linspace(f_min, f_max, M)
            s_params_partial = dict_to_matrix(self.s_parameters(wl=speed_of_light/f_partial))
            df = jnp.abs(f_partial[1] - f_partial[0])
            t = jnp.arange(-M//2, M//2) * 1/(M*df)
            h = jnp.fft.ifft(jnp.fft.ifftshift(jnp.conj(s_params_partial)), axis=0)
            h = jnp.fft.fftshift(h)

            h = h[M//2:, :, :]
            t = t[M//2:]
            
            
            max_energy_loss_percentage = 0.0001
            signal_energy_density = jnp.abs(h)**2
            signal_energy = jnp.sum(signal_energy_density, axis=(0))
            cumulative_signal_energy = jnp.cumsum(signal_energy_density, axis=0)
            mask = cumulative_signal_energy >= max_energy_loss_percentage*signal_energy
            delay_indices = jnp.argmax(mask, axis=0)
            delay = delay_indices*(t[1]-t[0])
            plt.plot(t, jnp.abs(h)[:, 0, 1])
            plt.axvline(delay[0, 1], color='r')
            plt.xlim(0, 20e-12)
            plt.xlabel("Time (s)")
            plt.ylabel("e-field amplitude")
            plt.show()
            
            bandwidth = f_max - f_min
            phase = jnp.unwrap(jnp.angle(s_params), axis=0)
            group_delay = jnp.gradient(phase, 2*jnp.pi*f, axis=0)
            avg_group_delay = jnp.abs(group_delay).mean(axis=0)
            max_group_delay = jnp.max(avg_group_delay)
            
            min_model_order_estimate = int(jnp.maximum(self.min_model_order, 2*bandwidth*max_group_delay))
            min_model_order_estimate = int(jnp.minimum(min_model_order_estimate, self.max_model_order))
            poles, residues, feedthrough, error = optimize_order_vector_fitting_discrete(min_model_order_estimate, self.max_model_order, s_params, f, f_c, f_s)
            
            if error > self.max_error:
                raise ValueError(f"Max Error Exceeded. Increase the model order or consider a different modeling strategy for {sax_model}")
            
            A, B, C, D = state_space_discrete(poles, residues, feedthrough)
            self.state_space_model = (A, B, C, D)
            self.center_frequency = f_c
            
            H = pole_residue_response_discrete(f, f_c, f_s, poles, residues, feedthrough)
            # H_full = pole_residue_response_discrete(jnp.linspace(-f_s/2, f_s/2, 1000)+f_c, f_c, f_s, poles, residues, feedthrough)
            # print(f"NUMBER OF POLES: {len(poles)}")
            # plt.plot(f, jnp.abs(H[:, 0, 1])**2)
            # plt.plot(f, jnp.abs(s_params[:, 0, 1])**2)
            # plt.show()
            # plt.plot(jnp.linspace(-f_s/2, f_s/2, 1000), jnp.abs(H_full[:, 0, 1])**2)
            # plt.show()
            time_step = 0
            x = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), A.shape[0]), dtype=complex)
            return time_step, x
        
        def sample_mode_step_optimal_order(
            self,
            inputs: dict,
            state: jax.Array,
            simulation_parameters,
        ):
            time_step, x = state
            A, B, C, D = self.state_space_model
            
            u = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), len(optical_ports)),dtype=complex)
            TE_MODE = 0
            for port, signal in inputs.items():
                port_idx = self.port_order[port]
                wavelength = inputs[port].wavelength
                u = u.at[:, port_idx].set(signal.amplitude[:, TE_MODE])
            
            new_x = jnp.zeros_like(x)
            y = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), len(optical_ports)),dtype=complex)

            for i, f in enumerate(speed_of_light/simulation_parameters.optical_baseband_wavelengths):
                detuning = 2*jnp.pi*(f-self.center_frequency)
                t = simulation_parameters.sampling_period * time_step
                new_x = new_x.at[i].set(jnp.exp(1j*detuning*simulation_parameters.sampling_period)*(A@x[i] + B@u[i]))
                y = y.at[i].set(jnp.exp(1j*detuning*self.delay_compensation*simulation_parameters.sampling_period)*(C@x[i] + D@u[i]))
                # y = jnp.exp(-1j*detuning*self.delay_compensation*simulation_parameters.sampling_period)*y

            outputs = {}
            for port in optical_ports:
                A_t = y[:, self.port_order[port]]
                outputs[port] = SampleModeOpticalSignal(
                    ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
                    amplitude = A_t.reshape((len(simulation_parameters.optical_baseband_wavelengths), 1)),
                    wavelength = simulation_parameters.optical_baseband_wavelengths
                )

            return outputs, (time_step + 1, new_x) 

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
