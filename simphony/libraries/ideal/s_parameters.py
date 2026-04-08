from jax.typing import ArrayLike
import jax.numpy as jnp
import jax

from sax import get_ports
from sax.saxtypes import Model

import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

import matplotlib.pyplot as plt
from simphony.time_domain.vector_fitting.z_domain import optimize_order_vector_fitting_discrete, vector_fitting_discrete, pole_residue_response_discrete, state_space_discrete, PHYSICIST
from simphony.signal.sample_mode import SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal
from simphony.signal.steady_state import SteadyStateOpticalSignal

from scipy.constants import speed_of_light

from simphony.component.port import Port
from simphony.component.component import SParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent
from simphony.utils import dict_to_matrix, dict_to_rect_matrix

from simphony.component.pcell import PCell

from simphony.libraries.ideal.digital_filters import OpticalDiscreteFilter
from simphony.libraries.ideal.modulators import OpticalModulator

from typing import Type

from simphony.libraries.ideal.digital_filters import discrete_state_space
from simphony.libraries.ideal.multimode import ModeConverter, mode_multiplexer, mode_demultiplexer

from simphony.simulation.simulation import SimulationMode, SimulationParameters
# from simphony.simulation.s_parameter import SParameterSimulation
# from simphony.simulation.sample_mode import SampleModeSimulation
# from simphony.simulation.block_mode import BlockModeSimulation

from sax import DEFAULT_MODES
from copy import deepcopy
# from simphony.circuit.netlist import InstantiatedFlatNetlist, instantiate_netlist
from simphony.circuit._netlist import InstantiatedFlatNetlist
from simphony.circuit._netlist import _instantiate_netlist
import warnings


INPUT_SUFFIX = "in"
OUTPUT_SUFFIX = "out"
MULTIPLEXER_SUFFIX = "_mux"
DEMULTIPLEXER_SUFFIX = "_demux"
DEMULTIPLEXER_IN_PORT_NAME = "in_port"
DEMULTIPLEXER_OUT_PORT_SUFFIX = "_port"
MULTIPLEXER_IN_PORT_SUFFIX = "_port"
MULTIPLEXER_OUT_PORT_NAME = "out_port"
MODE_CONVERTER_MODEL_NAME = "mode_converter"
MODE_CONVERTER_INSTANCE_SUFFIX = "_converter"
FIR_FILTER_MODEL_NAME = "fir_filter"
FIR_FILTER_INSTANCE_NAME = FIR_FILTER_MODEL_NAME

# TODO: CHANGE BLOCK MODE DESIGN SO THAT THE CODE REFLECTS HOW IT JUST USES A STATE SPACE MODEL
STATE_SPACE_MODEL_NAME = "fir_filter"
STATE_SPACE_INSTANCE_NAME = FIR_FILTER_MODEL_NAME

class SParameterSax(PCell):
    """
    Using the Component Factory Below
    """


def _default_vector_fitting_parameters(simulation_parameters):
    return {
        "spectral_range": simulation_parameters.spectral_range,
        "model_order": None,
        "min_model_order": 2,
        "max_model_order": 100,
        "num_frequency_samples": 1000,
        "center_frequency": speed_of_light / simulation_parameters.center_wavelength,
        "sampling_frequency": 1 / simulation_parameters.dt,
    }

def optical_s_parameter(
    sax_model: sax.Model, 
    port_directionality: dict = None,
    default_modes: list|tuple|str = DEFAULT_MODES,
)-> type[PCell]:
    """
    The directionality of each port defaults to 'bidirectional', 
    but individual ports may be set to 'bidirectional', 'input', or 'output
    by supplying a dictionary with port name keys.

    It is necessary for any directional simulators, such as the BlockModeSimulation class to specify the directionality of each port

    default_mode_identifier: since sax circuits do not require the user
    to specify the mode by default, we assign each relationship to the TE/TM mode by default (replicating behavior across the two different modes),
    if unspecified. Refer to sax.multimode for more details
    """
    pcell_port_names = _get_port_names_without_mode(sax_model)

    port_directionality = deepcopy(port_directionality)
    if port_directionality is None:
        port_directionality = {}

    for port_name in pcell_port_names:
        port_directionality.setdefault(port_name, "bidirectional")
    
    default_port_directionality = port_directionality


    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)
    
    BaseSParameterSax = SParameterSax # Freeze the reference
    class SpecificSParameterSax(BaseSParameterSax):
        ports = [
            Port(
                name=port_name,
                type="optical",
                directionality = port_directionality[port_name]
            ) 
            for port_name in pcell_port_names
        ]
        
        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            sax_settings: dict = None,
            vector_fitting_parameters = None,
            delay_compensation: int = 0,
            port_directionality = None,
        ):
            if port_directionality is None:
                port_directionality = default_port_directionality
            
            if sax_settings is None:
                sax_settings = {}
            
            # TODO: Update the Vector Fitting Code to have a dataclass for these parameters
            # TODO: Perhaps put vector fitting params in the simulation parameters as a "default_vector_fitting_params" field
            default_vector_fitting_parameters = _default_vector_fitting_parameters(simulation_parameters)
            if vector_fitting_parameters is None:
                vector_fitting_parameters = default_vector_fitting_parameters
            else:
                vector_fitting_parameters = default_vector_fitting_parameters | vector_fitting_parameters


            designs = {
                SimulationMode.S_PARAMETER: _s_parameter_design,
                SimulationMode.BLOCK_MODE: _block_mode_design,
                SimulationMode.SAMPLE_MODE: _sample_mode_design,
            }

            self.netlist, self.models, self.settings = designs[simulation_parameters.simulation_mode](sax_model, sax_settings, simulation_parameters, vector_fitting_parameters, delay_compensation, port_directionality)
    
    return SpecificSParameterSax

def _s_parameter_design(
    sax_model: sax.Model, 
    sax_settings, 
    simulation_parameters,
    vector_fitting_parameters,
    delay_compensation, 
    port_directionality, 
    # default_modes
):
    """
    `port_directionality`: As of 03-16-26, Simphony makes a "best guess" for sax models that have ambiguous port directionality.
    For best results, convert sax models to a Simphony Component first
    TODO: Link to a tutorial on how to convert sax models to simphony Components
    """
    if not delay_compensation == 0:
        warnings.warn(f"A nonzero delay compensation is invalid in S-Parameter simulations. Will be ignored.")
    
    # TODO: PROPERLY FILTER SAX MODEL WITH THE DEFAULT MODES SPECIFIED in simulation_parameters.mode_identifiers
    class SaxModelComponent(SParameterComponent):
        ports = [
            Port(
                name=port_name,
                type="optical",
                directionality = "bidirectional"
            )

            for port_name in sax.get_ports(sax_model())
        ]

        def __init__(
            self,
            simulation_mode: SimulationMode,
            **kwargs,
        ):
            self.sax_settings = kwargs
        
        def s_parameters(self, inputs, wl: ArrayLike=1.55e-6,):
            return sax_model(wl*1e6, **sax_settings)

    
    # Create a class that will be the base component
    # Extend s-paraemeters to all of the default_modes using sax
    instances = {
        "sax_model": "sax_model",
    }
    connections = {}
    ports = {port.name:f"sax_model,{port.name}" for port in SaxModelComponent.ports}
    models = {
        "sax_model": SaxModelComponent,
    }

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports,
    }

    settings = {
        "sax_model": {**sax_settings}
    }
    
    return netlist, models, settings


def _sample_mode_design(
    sax_model: sax.Model, 
    sax_settings, 
    simulation_parameters,
    vector_fitting_parameters,
    delay_compensation, 
    port_directionality, 
    # default_modes
):
    """
    The sample mode design is an alteration of the block mode design
    """
    block_mode_sax_model, block_mode_port_directionality, in_suffix, out_suffix = _bidirectional_ports_to_unidirectional_ports(sax_model, port_directionality)
    block_mode_netlist, block_mode_models = _block_mode_netlist_and_models(block_mode_sax_model, block_mode_port_directionality, simulation_parameters.mode_identifiers)
    
    netlist = ...
    # TODO: Write a function that takes in settings and models and returns instantiated models
    instantiated_models = ...

    return netlist, instantiated_models 

def _block_mode_design(
    sax_model: sax.ModelMM, 
    sax_settings, 
    simulation_parameters,
    vector_fitting_parameters,
    delay_compensation, 
    port_directionality, 
    # default_modes
)->InstantiatedFlatNetlist:
    """
    Creates a directed, non recursive circuit design for a multimodal,
    transient block mode simulation of an s-parameter circuit.
    """
    print(port_directionality)
    for port_name, directionality in port_directionality.items():
        if directionality == "bidirectional":
            raise ValueError(f"{sax_model}'s Port {port_name} must be directed in block mode simulation")
    
    if not delay_compensation == 0:
        warnings.warn(f"A nonzero delay compensation is invalid in block mode simulations. Will be ignored.")
    
    filtered_sax_model = _get_filtered_sax_model(sax_model, port_directionality, simulation_parameters.mode_identifiers)
    netlist, models = _block_mode_netlist_and_models(filtered_sax_model)

    input_port_modes, output_port_modes = _get_port_mode_luts(filtered_sax_model)

    print(f"Building Model For {sax_model}")
    
    f_min = speed_of_light / max(vector_fitting_parameters['spectral_range'])
    f_max = speed_of_light / min(vector_fitting_parameters['spectral_range'])
    f_center = vector_fitting_parameters['center_frequency']
    # f_center = 192.9e12
    frequency = jnp.linspace(f_min, f_max, vector_fitting_parameters["num_frequency_samples"])
    s_params = dict_to_rect_matrix(filtered_sax_model(wl=1e6*speed_of_light/frequency, **sax_settings), input_ports=[f"{port}@{mode}" for port, modes in input_port_modes.items() for mode in modes], output_ports=[f"{port}@{mode}" for port, modes in output_port_modes.items() for mode in modes])
    min_order = vector_fitting_parameters["min_model_order"]
    max_order = vector_fitting_parameters["max_model_order"]
    sampling_frequency = vector_fitting_parameters["sampling_frequency"]
    
    ### TODO: REMOVE THIS LINE USED FOR TESTING
    # vector_fitting_parameters["model_order"] = 20
    
    if vector_fitting_parameters["model_order"] is None:
        poles, residues, feedthrough, mean_squared_error = optimize_order_vector_fitting_discrete(min_order, max_order, s_params, frequency, f_center, sampling_frequency, sign_convention=PHYSICIST)
    else:
        poles, residues, feedthrough, mean_squared_error = vector_fitting_discrete(vector_fitting_parameters["model_order"], s_params, frequency, f_center, sampling_frequency, sign_convention=PHYSICIST)

    A, B, C, D = state_space_discrete(poles, residues, feedthrough)

    ### TODO: Fill in empty settings..s
    common_mode = simulation_parameters.mode_identifiers[0]
    settings = {}
    settings.update({FIR_FILTER_INSTANCE_NAME:{"A":A, "B":B, "C":C, "D":D}})
    settings.update({_mode_converter_instance_name(port, mode, INPUT_SUFFIX):{"input_mode":mode,"output_mode": common_mode} for port, modes in input_port_modes.items() for mode in modes})
    settings.update({_mode_converter_instance_name(port, mode, OUTPUT_SUFFIX):{"input_mode":common_mode, "output_mode": mode} for port, modes in output_port_modes.items() for mode in modes})
    settings.update({_demultiplexer_instance_name(port):{} for port in input_port_modes.keys()})
    settings.update({_multiplexer_instance_name(port):{} for port in output_port_modes.keys()})
    
    # instantiated_flat_netlist = _instantiate_netlist(netlist, models, settings)

    return netlist, models, settings

def _block_mode_netlist_and_models(
    filtered_sax_model: sax.Model
):
    """
    Returns a dicts defining the instances, connections, and ports of the subcircuit
    as well as a dict of uninstantiated models
    """
    connections = {}
    instances = {}
    ports = {}
    models = {}

    input_port_modes, output_port_modes = _get_port_mode_luts(filtered_sax_model)

    mode_demultiplexers = {}
    fir_filter_input_port_names = []
    for input_port, modes in input_port_modes.items():        
        mode_demultiplexers[input_port] = mode_demultiplexer(modes, input_port_name=DEMULTIPLEXER_IN_PORT_NAME,output_port_suffix=DEMULTIPLEXER_OUT_PORT_SUFFIX)
        fir_filter_input_port_names += [_fir_filter_port_name(input_port, mode) for mode in modes]

    mode_multiplexers = {}
    fir_filter_output_port_names = []
    for output_port, modes in output_port_modes.items():
        mode_multiplexers[output_port] = mode_multiplexer(modes, output_port_name=MULTIPLEXER_OUT_PORT_NAME, input_port_suffix=MULTIPLEXER_IN_PORT_SUFFIX)
        fir_filter_output_port_names +=[_fir_filter_port_name(output_port, mode) for mode in modes]


    # TODO: CHANGE THIS CODE TO REFLECT THE DISCRETESTATESPACEMODEL
    FIRFilter = discrete_state_space(
        len(fir_filter_input_port_names), 
        len(fir_filter_output_port_names),
        fir_filter_input_port_names,
        fir_filter_output_port_names,
    )

    models[MODE_CONVERTER_MODEL_NAME] = ModeConverter
    models[FIR_FILTER_MODEL_NAME] = FIRFilter
    instances[FIR_FILTER_INSTANCE_NAME] = FIR_FILTER_MODEL_NAME 

    # Input Side Demultiplexers and Mode Converters
    for port, demux in mode_demultiplexers.items():
        demux_model_name = _demultiplexer_model_name(port)
        models[demux_model_name] = demux
        demux_instance_name = _demultiplexer_instance_name(port)
        instances[demux_instance_name] = demux_model_name

        modes = input_port_modes[port]
        for mode in modes:
            mode_converter_instance_name = _mode_converter_instance_name(port, mode, INPUT_SUFFIX)
            instances[mode_converter_instance_name] = MODE_CONVERTER_MODEL_NAME
            demux_output = demux_instance_name + "," + mode + DEMULTIPLEXER_OUT_PORT_SUFFIX
            converter_input = mode_converter_instance_name + ',' + 'in'
            converter_output = mode_converter_instance_name + ',' + 'out'
            fir_filter_input = FIR_FILTER_INSTANCE_NAME + ',' + _fir_filter_port_name(port, mode)
            connections[demux_output] = converter_input
            connections[converter_output] = fir_filter_input
        
        ports[port] = demux_instance_name + "," + DEMULTIPLEXER_IN_PORT_NAME
    
    # Output Side Multiplexers and Mode Converters
    for port, mux in mode_multiplexers.items():
        mux_model_name = _multiplexer_model_name(port)
        models[mux_model_name] = mux
        mux_instance_name = _multiplexer_instance_name(port)
        instances[mux_instance_name] = mux_model_name

        modes = output_port_modes[port]
        for mode in modes:
            mode_converter_instance_name = _mode_converter_instance_name(port, mode, OUTPUT_SUFFIX)
            instances[mode_converter_instance_name] = MODE_CONVERTER_MODEL_NAME
            mux_input = mux_instance_name + "," + mode + MULTIPLEXER_IN_PORT_SUFFIX
            converter_input = mode_converter_instance_name + ',' + 'in'
            converter_output = mode_converter_instance_name + ',' + 'out'
            fir_filter_output = FIR_FILTER_INSTANCE_NAME + ',' + _fir_filter_port_name(port, mode)
            
            connections[fir_filter_output] = converter_input
            connections[converter_output] = mux_input

        ports[port] = mux_instance_name + "," + MULTIPLEXER_OUT_PORT_NAME

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports,
    }
    
    return netlist, models

def _mode_converter_instance_name(
    port,
    mode,
    direction
)->str:
    return port + '_' + mode + MODE_CONVERTER_INSTANCE_SUFFIX + '_' + direction

def _demultiplexer_model_name(
    port,
):
    return port + DEMULTIPLEXER_SUFFIX

def _demultiplexer_instance_name(
    port,
):
    return _demultiplexer_model_name(port)

def _multiplexer_model_name(
    port,
):
    return port + MULTIPLEXER_SUFFIX

def _multiplexer_instance_name(
    port,
):
    return _multiplexer_model_name(port)

def _fir_filter_port_name(
    port,
    mode,
):
    return port + "_" + mode

def _get_filtered_sax_model(
    sax_model: sax.Model,
    port_directionality,
    default_modes,
):
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

    return filtered_sax_model

def _get_port_mode_luts(
    sax_model: sax.Model,
):
    input_port_modes = {}
    output_port_modes = {}
    for o, i in sax_model().keys():
        in_port, in_mode = i.split('@')
        out_port, out_mode = o.split('@')
        input_port_modes.setdefault(in_port, set()).add(in_mode)
        output_port_modes.setdefault(out_port, set()).add(out_mode)

    return input_port_modes, output_port_modes

def _bidirectional_ports_to_unidirectional_ports(
    sax_model: sax.ModelMM, 
    port_directionality
):
    """
    The resulting port names are the orginal port names appended 
    with a string unique to all substrings in the original port names.

    In order to find the original port name, just remove the 
    out_suffix and in_suffix substrings from the dict keys.
    """
    port_names = port_directionality.keys()
    unique_word = "_"

    for port_name in port_names:
        while(unique_word in port_name):
            unique_word += "_"
    
    in_suffix = unique_word + INPUT_SUFFIX
    out_suffix = unique_word + OUTPUT_SUFFIX
    

    unidirectional_port_directionality = {}
    valid_input_ports = set()
    valid_output_ports = set()
    for port_name, directionality in port_directionality.items():
        in_port_name = port_name + in_suffix
        out_port_name = port_name + out_suffix
        if directionality == "bidirectional":
            unidirectional_port_directionality[in_port_name] = "input"
            unidirectional_port_directionality[out_port_name] = "output"
            valid_input_ports.add(in_port_name)
            valid_output_ports.add(out_port_name)
        elif directionality == "input":
            # in_port_name = port_name + f"{unique_word}in"
            unidirectional_port_directionality[in_port_name] = "input"
            valid_input_ports.add(in_port_name)
        elif directionality == "output":
            # out_port_name = port_name + f"{unique_word}out"
            unidirectional_port_directionality[out_port_name] = "output"
            valid_output_ports.add(out_port_name)

    def unidirectional_sax_model(**kwargs):
        valid_input_ports
        valid_output_ports
        in_suffix
        out_suffix
        sdict = sax_model(**kwargs)
        new_sdict = {(src+out_suffix, dst+in_suffix): v for (src, dst), v in sdict.items()}

        return new_sdict

    return unidirectional_sax_model, unidirectional_port_directionality, in_suffix, out_suffix 
    

def _get_port_names_without_mode(sax_model):
    sdict = sax.multimode(sax_model())
    port_names = set()
    for in_portmode, out_portmode in sdict.keys():
        in_port, _ = in_portmode.split('@')
        out_port, _ = out_portmode.split('@')
        port_names.add(in_port)
        port_names.add(out_port)
    
    return port_names

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

