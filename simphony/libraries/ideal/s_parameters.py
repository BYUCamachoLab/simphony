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

from simphony.component.pcell import PCell

from simphony.libraries.ideal.filters import OpticalDiscreteFilter
from simphony.libraries.ideal.modulators import OpticalModulator

from typing import Type

from simphony.libraries.ideal.filters import discrete_state_space
from simphony.libraries.ideal.multimode import ModeConverter, mode_multiplexer, mode_demultiplexer

from sax import DEFAULT_MODES

_s_parameter_netlist = {

}

_s_parameter_models = {

}

def optical_s_parameter(
    sax_model: SaxModel, 
    port_directionality = None,
    default_modes: list|tuple|str = DEFAULT_MODES,
)-> type[PCell]:
    """
    The directionality of each port defaults to 'bidirectional', 
    but individual ports may be set to 'bidirectional', 'input', or 'output
    by supplying a dictionary with port name keys.

    default_mode_identifier: since sax circuits do not require the user
    to specify the mode by default, we assign each relationship to the TE/TM mode by default (replicating behavior across the two different modes),
    if unspecified. Refer to sax.multimode for more details
    """
    
    ### TODO: Implement Multimodal Collapse ###
    # 1) Determining all of the surviving modes (which modes are contained in the s-parameter matrix)
    # 2) Create a MIMO system with all of the mode relationships
    # 3) Shift all modes to the same mode
    ###########################################

    if port_directionality is None:
        port_directionality = {}
    
    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)

    input_ports_to_remove = {port_name for port_name, direction in port_directionality.items() if direction=='output'}
    output_ports_to_remove = {port_name for port_name, direction in port_directionality.items() if direction=='input'}

    def filtered_sax_model(**kwargs):
        """
        The port_directionality field allows us to ignore data
        in the sdict and select only the relationships necessary
        for the specified directionality
        """
        sdict = sax_model(**kwargs)
        sdict = sax.multimode(sdict, modes=default_modes)

        def is_allowed(key):
            dst, src = key
            src_port, _ = src.split("@")
            dst_port, _ = dst.split("@")
            src_valid = not src_port in input_ports_to_remove
            dst_valid = not dst_port in output_ports_to_remove
            return src_valid and dst_valid

        sdict = {k:v for k, v in sdict.items() if is_allowed(k)}

        return sdict


    # As of sax 0.15.10, get_modes does not necessarily return a tuple of UNIQUE values
    # modes = tuple(set(sax.get_modes(sax_model())))

    # Sax does not require every mode relation specified (I think that the port
    # to port relations default to 0 in that case)
    # port_modes = {}
    # for p in sax.get_ports(sax_model()):
    #     port, mode = p.split("@")
    #     port_modes.setdefault(port, set()).add(mode)
    
    input_port_modes = {}
    output_port_modes = {}
    for i, o in filtered_sax_model().keys():
        in_port, in_mode = i.split('@')
        out_port, out_mode = o.split('@')
        input_port_modes.setdefault(in_port, set()).add(in_mode)
        output_port_modes.setdefault(out_port, set()).add(out_mode)
    
    pcell_port_names = list(set(input_port_modes.keys()) | set(output_port_modes.keys()))

    class SParameterSax(PCell):
        ports = [
            Port(
                name=port_name, 
                type="optical", 
                directionality = port_directionality.get(port_name, 'bidirectional')
            ) 
            for port_name in pcell_port_names
        ]
        

        def __init__(
            self,
            spectral_range: tuple = (1.5e-6, 1.6e-6),
            delay_compensation: int = 0,
            sax_settings: dict = {},
        ):
            mode_demultiplexers = {}
            num_filter_inputs = 0
            for input_port, modes in input_port_modes.items():
                mode_demultiplexers[input_port] = mode_demultiplexer(modes)
                num_filter_inputs += len(modes)

            mode_multiplexers = {}
            num_filter_outputs = 0
            for output_port, modes in output_port_modes.items():
                mode_multiplexers[output_port] = mode_multiplexer(modes)
                num_filter_outputs += len(modes)
            
            FIRFilter = discrete_state_space(
                num_filter_inputs, 
                num_filter_outputs,
                [],
                [],
            )

            models = {}
            models["converter"] = ModeConverter
            models["fir_filter"] = FIRFilter
            models["phase_shifter"] = OpticalModulator
            
            # TODO: Connect Phase Shifters to muxes
            for port_name, mux in mode_multiplexers.items():
                mux_name = f"{port_name}_{mux}"
                mux_instance_name = mux_name
                models[mux_name] = mux
                instances[mux_instance_name] = mux_name
                new_connections = {
                    f"{},{}":f"{mux_instance_name},{mode}"
                } 

            for port_name, demux in mode_demultiplexers.items():
                models[f"{port_name}_{demux}"] = demux


            instances = {}
            connections = {}
            
            input_ports = {port_name:f"{1},{2}" for port_name in pcell_port_names}
            ports = {}

            netlist = {
                "instances": instances,
                "connections": connections,
                "ports": ports,
            }
            

            self.netlist = netlist
            self.models = models

    
    return SParameterSax


# def optical_s_parameter(sax_model: SaxModel):
#     optical_ports = list(sax.get_ports(sax_model()))
#     class SParameterSax(OpticalSParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent):
#         ports = [
#             Port(name=port_name, type="optical", directionality="bidirectional") for port_name in optical_ports
#         ]
#         _num_ports = len(ports)
        
#         def __init__(
#             self, 
#             spectral_range=(1.5e-6,1.6e-6),
#             delay_compensation=0,
#             max_error=1e-6,
#             min_model_order=45,
#             max_model_order=50,
#             method = 'optimal_order',
#             **sax_settings
#         ):
#             # super().__init__(**settings)
#             self.max_error = max_error
#             self.min_model_order = min_model_order
#             self.max_model_order = max_model_order
#             self.delay_compensation = delay_compensation
#             self.settings = sax_settings
#             self.spectral_range = spectral_range
#             self.port_order = {name: idx for idx, name in enumerate(optical_ports)}

#             if method == 'optimal_order':
#                 self.sample_mode_initial_state = self.sample_mode_initial_state_optimal_order
#                 self.sample_mode_step = self.sample_mode_step_optimal_order

#         def sample_mode_initial_state_optimal_order(
#             self,
#             simulation_parameters,
#         ):
#             max_group_delay = 10e-12
#             N = 1000
#             f_min = speed_of_light / self.spectral_range[1]
#             f_max = speed_of_light / self.spectral_range[0]
            
#             f = jnp.linspace(f_min, f_max, N)
#             f_c = 0.5*(f_max + f_min)
#             f_s = 1 / simulation_parameters.sampling_period
#             s_params = dict_to_matrix(self.s_parameters(wl=speed_of_light/f))
            
#             M = 10000
#             f_partial = jnp.linspace(f_min, f_max, M)
#             s_params_partial = dict_to_matrix(self.s_parameters(wl=speed_of_light/f_partial))
#             df = jnp.abs(f_partial[1] - f_partial[0])
#             t = jnp.arange(-M//2, M//2) * 1/(M*df)
#             h = jnp.fft.ifft(jnp.fft.ifftshift(jnp.conj(s_params_partial)), axis=0)
#             h = jnp.fft.fftshift(h)

#             h = h[M//2:, :, :]
#             t = t[M//2:]
            
            
#             max_energy_loss_percentage = 0.0001
#             signal_energy_density = jnp.abs(h)**2
#             signal_energy = jnp.sum(signal_energy_density, axis=(0))
#             cumulative_signal_energy = jnp.cumsum(signal_energy_density, axis=0)
#             mask = cumulative_signal_energy >= max_energy_loss_percentage*signal_energy
#             delay_indices = jnp.argmax(mask, axis=0)
#             delay = delay_indices*(t[1]-t[0])
#             plt.plot(t, jnp.abs(h)[:, 0, 1])
#             plt.axvline(delay[0, 1], color='r')
#             plt.xlim(0, 20e-12)
#             plt.xlabel("Time (s)")
#             plt.ylabel("e-field amplitude")
#             plt.show()
            
#             bandwidth = f_max - f_min
#             phase = jnp.unwrap(jnp.angle(s_params), axis=0)
#             group_delay = jnp.gradient(phase, 2*jnp.pi*f, axis=0)
#             avg_group_delay = jnp.abs(group_delay).mean(axis=0)
#             max_group_delay = jnp.max(avg_group_delay)
            
#             min_model_order_estimate = int(jnp.maximum(self.min_model_order, 2*bandwidth*max_group_delay))
#             min_model_order_estimate = int(jnp.minimum(min_model_order_estimate, self.max_model_order))
#             poles, residues, feedthrough, error = optimize_order_vector_fitting_discrete(min_model_order_estimate, self.max_model_order, s_params, f, f_c, f_s)
            
#             if error > self.max_error:
#                 raise ValueError(f"Max Error Exceeded. Increase the model order or consider a different modeling strategy for {sax_model}")
            
#             A, B, C, D = state_space_discrete(poles, residues, feedthrough)
#             self.state_space_model = (A, B, C, D)
#             self.center_frequency = f_c
            
#             H = pole_residue_response_discrete(f, f_c, f_s, poles, residues, feedthrough)
#             time_step = 0
#             x = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), A.shape[0]), dtype=complex)
#             return time_step, x
        
#         def sample_mode_step_optimal_order(
#             self,
#             inputs: dict,
#             state: jax.Array,
#             simulation_parameters,
#         ):
#             time_step, x = state
#             A, B, C, D = self.state_space_model
            
#             u = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), len(optical_ports)),dtype=complex)
#             TE_MODE = 0
#             for port, signal in inputs.items():
#                 port_idx = self.port_order[port]
#                 wavelength = inputs[port].wavelength
#                 u = u.at[:, port_idx].set(signal.amplitude[:, TE_MODE])
            
#             new_x = jnp.zeros_like(x)
#             y = jnp.zeros((len(simulation_parameters.optical_baseband_wavelengths), len(optical_ports)),dtype=complex)

#             for i, f in enumerate(speed_of_light/simulation_parameters.optical_baseband_wavelengths):
#                 detuning = 2*jnp.pi*(f-self.center_frequency)
#                 t = simulation_parameters.sampling_period * time_step
#                 new_x = new_x.at[i].set(jnp.exp(1j*detuning*simulation_parameters.sampling_period)*(A@x[i] + B@u[i]))
#                 y = y.at[i].set(jnp.exp(1j*detuning*self.delay_compensation*simulation_parameters.sampling_period)*(C@x[i] + D@u[i]))
#                 # y = jnp.exp(-1j*detuning*self.delay_compensation*simulation_parameters.sampling_period)*y

#             outputs = {}
#             for port in optical_ports:
#                 A_t = y[:, self.port_order[port]]
#                 outputs[port] = SampleModeOpticalSignal(
#                     ### TODO: FIX THIS FOR MULTIPLE POLARIZATIONS/MODES
#                     amplitude = A_t.reshape((len(simulation_parameters.optical_baseband_wavelengths), 1)),
#                     wavelength = simulation_parameters.optical_baseband_wavelengths
#                 )

#             return outputs, (time_step + 1, new_x) 

#         # @staticmethod
#         # @jax.jit
#         def s_parameters( 
#             self,
#             inputs: dict=None,
#             wl: ArrayLike=1.55e-6,
#         )->sax.SDict:
#             # TODO: (MATTHEW! Don't do this one yet, I need to talk to Sequoia first)
#             # Change the simphony models to be in units of meters not microns 
#             return sax_model(wl*1e6, **self.settings)
        
#         # @staticmethod
#         # @jax.jit 
#         def steady_state(self, inputs: dict):
#             # Sadly, sax_model is not jit compatible
#             # so instead we just jit what we can.
#             # complete_steady_state_inputs(inputs)
#             ports = sax.get_ports(sax_model)
#             wl = inputs[ports[0]].wl
#             s_params = dict_to_matrix(sax_model(wl*1e6, **self.settings))
#             outputs = self._compute_outputs(s_params, wl, inputs)
            
#             return outputs
        
#         @staticmethod
#         @jax.jit
#         def _compute_outputs(s_params: ArrayLike, wls, inputs:dict)->dict:
#             ports = sax.get_ports(sax_model)
#             num_ports = len(ports)
#             num_wls = wls.shape[0]
#             input_matrix = jnp.zeros((num_wls, num_ports), dtype=complex)
#             for i, port in enumerate(ports):
#                 input_matrix = input_matrix.at[:, i].set(inputs[port].field)
            
#             output_matrix = jnp.zeros_like(input_matrix)
#             for i, wl in enumerate(wls):
#                 _output = s_params[i,:,:] @ input_matrix[i, :]
#                 output_matrix = output_matrix.at[i, :].set(_output)

#             outputs = {}
#             for i, port in enumerate(ports):
#                 outputs[port] = SteadyStateOpticalSignal(
#                                     field=output_matrix[:, i],
#                                     wl=wls,
#                                     polarization=inputs[port].polarization
#                                 )
            
#             return outputs

#     return SParameterSax


