from jax.typing import ArrayLike
import jax.numpy as jnp
import jax

from sax import circuit

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

from simphony.component.placeholder import Placeholder
from simphony.circuit.netlist import sanitize_instance_names, generate_valid_separator, netlist_to_graph, add_settings_to_netlist
import inspect

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
STATE_SPACE_MODEL_NAME_BASE = "state_space"
STATE_SPACE_INSTANCE_NAME_BASE = STATE_SPACE_MODEL_NAME_BASE

_default_vector_fitting_parameters = {
                "model_order": None,
                "min_model_order": 2,
                "max_model_order": 100,
                "num_frequency_samples": 1000,
                "center_wavelength": 1.55e-6,
                "spectral_range": (1.5e-6, 1.6e-6),
                # NOTE: Currently, a user CAN change the spectral range parameter (the default is set by )
            }

class SParameterElement(SParameterComponent, SampleModeComponent):
    """
    The following component factory should be implemented when writing a simulator that interprets s-parameter elements using
    the default settings in the SParameterGroup PCell
    
    By default, SParameterGroup will expand each element within it into this model. If more
    flexibility and control is needed, feel free to write your own design in SParameterGroup
    """

def optical_s_parameter(
    sax_model: sax.Model, 
    port_directionality: dict = None,
    default_modes: list|tuple|str = DEFAULT_MODES,
)-> type[SParameterElement]:
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
    
    # default_port_directionality = port_directionality


    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)
    
    BaseSParameterElement = SParameterElement # Freeze the instance
    class SpecificSParameterElement(BaseSParameterElement):
        _sax_model = staticmethod(sax_model)
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
            # Rename kwargs to be more accurate
            **kwargs,
            # sax_settings: dict = None,
            # vector_fitting_parameters = None,
            # delay_compensation: int = 0,
            # port_directionality = {},
        ):
            """
            TODO: Add documentation for each of the kwargs
            """
            self.sax_model = sax_model
            self.settings = kwargs
            self.settings.setdefault('sax_settings', {})
            self.settings.setdefault('group_id', None)
            self.settings.setdefault('vector_fitting_parameters', _default_vector_fitting_parameters)
            self.settings.setdefault('delay_compensation', 0)
            self.settings.setdefault('port_directionality', {})
            # self.sax_model = sax_model
            # self.sax_settings = self.settings.setdefault('sax_settings', {})
            # self.vector_fitting_parameters = self.settings.setdefault('vector_fitting_parameters', default_vector_fitting_parameters)
            # self.delay_compensation = self.settings.setdefault('delay_compensation', 0)
            # self.port_directionality = self.settings.setdefault('port_directionality', {})
        
        def s_parameters(
            self,
            inputs: dict,
            wl: ArrayLike=1.55e-6,
        ):
            return sax_model(wl=wl, **self.settings["sax_settings"])

        def sample_mode_initial_state(self, simulation_parameters):
            """
            May be overwritten by user.
            Returns the initial the state of the system.
            Called by the sample mode simulator after `set_sample_mode_simulation_parameters`
            """
            return 0

        def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters):
            """Compute the next state of the system."""
            raise NotImplementedError
    
    return SpecificSParameterElement


# def sax_model_to_component(
#     sax_model: sax.Model, 
#     port_directionality: dict = None,
#     default_modes: list|tuple|str = DEFAULT_MODES,
# ):
#     port_names = _get_port_names_without_mode(sax_model)

#     port_directionality = deepcopy(port_directionality)
#     if port_directionality is None:
#         port_directionality = {}

#     for port_name in port_names:
#         port_directionality.setdefault(port_name, "bidirectional")
    
#     # default_port_directionality = port_directionality


#     if isinstance(default_modes, str):
#         default_modes = [default_modes]

#     default_modes = tuple(default_modes)
#     BaseSParameterElement = SParameterElement
#     class SpecificSParameterElement(BaseSParameterElement):

#         def s_parameters(
#             self,
#             inputs: dict,
#             wl: ArrayLike=1.55e-6,
#         ):
#             pass

#         def sample_mode_initial_state(self, simulation_parameters):
#             """
#             May be overwritten by user.
#             Returns the initial the state of the system.
#             Called by the sample mode simulator after `set_sample_mode_simulation_parameters`
#             """
#             return 0

#         def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters):
#             """Compute the next state of the system."""
#             raise NotImplementedError

class SParameterGroup(PCell):
    """
    Using the Component Factory Below
    """

def s_parameter_netlist_to_pcell(
    netlist: dict,
    models: dict,
    port_directionality: dict = None,
    default_modes: list|tuple|str = DEFAULT_MODES,
    fuse_models: bool = False,
) -> type[PCell]:
    """
    the models parameter should be structured like a sax netlist, but including a Simphony SParameter PCell is also fine
    The directionality of each port defaults to 'bidirectional', 
    but individual ports may be set to 'bidirectional', 'input', or 'output
    by supplying a dictionary with port name keys.

    It is necessary for any directional simulators, such as the BlockModeSimulation class to specify the directionality of each port

    default_mode_identifier: since sax circuits do not require the user
    to specify the mode by default, we assign each relationship to the TE/TM mode by default (replicating behavior across the two different modes),
    if unspecified. Refer to sax.multimode for more details
    
    Models will be fused according to their group_id
    """
    # pcell_port_names = _get_port_names_without_mode(sax_model)

    
    pcell_port_names = netlist['ports'].keys()
    def _get_pcell_port_directionality(port_directionality):
        pcell_port_directionality = {}
        for port_name in pcell_port_names:
            true_instance_name, true_port_name = netlist['ports'][port_name].split(',')
            pcell_port_directionality[port_name] = port_directionality[true_instance_name][true_port_name]
        
        return pcell_port_directionality
    
    pcell_port_directionality = _get_pcell_port_directionality(port_directionality)

    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)

    default_port_directionality = port_directionality
    default_pcell_port_directionality = pcell_port_directionality

    # Freeze the instances
    netlist = netlist
    models = models
    # settings = settings


    _SParameterGroup = SParameterGroup # Freeze the reference
    class SpecificSParameterGroup(_SParameterGroup):
        ports = [
            Port(
                name=port_name,
                type="optical",
                directionality = pcell_port_directionality[port_name]
            ) 
            for port_name in pcell_port_names
        ]
        
        def __init__(
            self,
            simulation_parameters: SimulationParameters,
            settings: dict = None,
            port_directionality: dict = None, # TODO: Figure out if this needs to be here
        ):
            if port_directionality is None:
                port_directionality = default_port_directionality
                pcell_port_directionality = default_pcell_port_directionality
            else:
                pcell_port_directionality = _get_pcell_port_directionality(port_directionality)
            
            
            # TODO: Make the way to edit these parameters more clear (add documentation to SParameterPlaceholder and how to use sax models)
            # TODO: Allow different models to edit "spectral_range" (by overwriting the simulation parameters)
            # TODO: Perhaps models can't overwrite simulation parameters spectral range, but can add additional pockets
            # "spectral_range": simulation_parameters.spectral_range,
            # "center_frequency": speed_of_light / simulation_parameters.center_wavelength,
            # "sampling_frequency": 1 / simulation_parameters.dt

            designs = {
                # SimulationMode._SIMPHONY_PREPROCESSING: _simphony_preprocessing,
                # SimulationMode.S_PARAMETER: _s_parameter_design,
                SimulationMode.BLOCK_MODE: _block_mode_design,
                # SimulationMode.SAMPLE_MODE: _sample_mode_design,
            }

            
            for _settings in settings.values():
                if not fuse_models:
                    _settings["group_id"] = None
                # This normalization should have been done previously
                # _settings.set("vector_fitting_parameters", default_vector_fitting_parameters)
                # _settings.setdefault("sax_settings", {})

                pass

            # TODO: Implement the fusing based on group id

            if simulation_parameters.simulation_mode in designs:
                design = designs[simulation_parameters.simulation_mode]
            else:
                design = _default_design

            internal_port_directionality = port_directionality
            external_port_directionality = pcell_port_directionality
            self.netlist, self.models, self.settings = design(netlist, models, settings, simulation_parameters, internal_port_directionality, external_port_directionality)
            
    return SpecificSParameterGroup

def _get_port_names_without_mode(sax_model):
    sdict = sax.multimode(sax_model())
    port_names = set()
    for in_portmode, out_portmode in sdict.keys():
        in_port, _ = in_portmode.split('@')
        out_port, _ = out_portmode.split('@')
        port_names.add(in_port)
        port_names.add(out_port)
    
    return port_names

# def _s_parameter_design(
#     netlist,
#     models,
#     settings,
#     simulation_parameters,
#     internal_port_directionality, 
#     external_port_directionality,
# ):
#     """
#     `port_directionality`: As of 03-16-26, Simphony makes a "best guess" for sax models that have ambiguous port directionality.
#     For best results, convert sax models to a Simphony Component first
#     TODO: Link to a tutorial on how to convert sax models to simphony Components
#     """
#     # if not delay_compensation == 0:
#     #     warnings.warn(f"A nonzero delay compensation is invalid in S-Parameter simulations. Will be ignored.")
    
#     # TODO: PROPERLY FILTER SAX MODEL WITH THE DEFAULT MODES SPECIFIED in simulation_parameters.mode_identifiers
#     class SaxModelComponent(SParameterComponent):
#         ports = [
#             Port(
#                 name=port_name,
#                 type="optical",
#                 directionality = "bidirectional"
#             )

#             for port_name in sax.get_ports(sax_model())
#         ]

#         def __init__(
#             self,
#             simulation_mode: SimulationMode,
#             **kwargs,
#         ):
#             self.sax_settings = kwargs
        
#         def s_parameters(self, inputs, wl: ArrayLike=1.55e-6,):
#             return sax_model(wl*1e6, **sax_settings)

    
#     # Create a class that will be the base component
#     # Extend s-paraemeters to all of the default_modes using sax
#     instances = {
#         "sax_model": "sax_model",
#     }
#     connections = {}
#     ports = {port.name:f"sax_model,{port.name}" for port in SaxModelComponent.ports}
#     models = {
#         "sax_model": SaxModelComponent,
#     }

#     netlist = {
#         "instances": instances,
#         "connections": connections,
#         "ports": ports,
#     }

#     settings = {
#         "sax_model": {**sax_settings}
#     }
    
#     return netlist, models, settings


# def _sample_mode_design(
#     netlist,
#     models,
#     settings,
#     simulation_parameters,
#     internal_port_directionality, 
#     external_port_directionality,
# ):
#     """
#     The sample mode design is an alteration of the block mode design
#     """
#     block_mode_sax_model, block_mode_port_directionality, in_suffix, out_suffix = _bidirectional_ports_to_unidirectional_ports(sax_model, port_directionality)
#     block_mode_netlist, block_mode_models = _block_mode_netlist_and_models(block_mode_sax_model, block_mode_port_directionality, simulation_parameters.mode_identifiers)
    
#     _netlist = ...
#     _models = ...

#     return _netlist, _models, settings 

def _default_design(
    netlist,
    models,
    original_settings,
    simulation_parameters,
    internal_port_directionality, 
    external_port_directionality,
):
    _netlist = netlist
    _models = {}
    for instance_name, instance_data in _netlist['instances'].items():
        model_name = instance_data['component']
        instance_data['component'] = instance_name
        model = models[model_name]
        _models[instance_name] = optical_s_parameter(model, internal_port_directionality[instance_name], simulation_parameters.mode_identifiers)
    # _models = {k:optical_s_parameter(v, internal_port_directionality[k], simulation_parameters.mode_identifiers) for k,v in models.items()}
    _settings = original_settings
    return _netlist, _models, _settings

def _block_mode_design(
    netlist,
    models,
    original_settings,
    simulation_parameters,
    internal_port_directionality, 
    external_port_directionality,
)->InstantiatedFlatNetlist:
    """
    Creates a directed, non recursive circuit design for a multimodal,
    transient block mode simulation of an s-parameter circuit.
    """
    for port_name, directionality in external_port_directionality.items():
        if directionality == "bidirectional":
            raise ValueError(f"Port {port_name} must be directed in block mode simulation")
    
    # if not delay_compensation == 0:
    #     warnings.warn(f"A nonzero delay compensation is invalid in block mode simulations. Will be ignored.")
    

    # TODO: This is really important
    # PROBLEM: The filtered sax models are not guarenteed to have the same directionality across instances
    # Solution: Make a new filtered_sax_model for each instance and update the netlist accordingly
    
    filtered_sax_models = {}
    filtered_port_designators = set()
    filtered_netlist = deepcopy(netlist)

    instance_names = filtered_netlist['instances'].keys()
    for instance_name in instance_names:
        model_name = filtered_netlist['instances'][instance_name]['component']
        model = models[model_name]
        port_directionality = internal_port_directionality[instance_name]
        filtered_sax_model = _get_filtered_sax_model(model, port_directionality, simulation_parameters.mode_identifiers)
        filtered_sax_models[instance_name] = filtered_sax_model
        filtered_port_designators.update([f"{instance_name},{p}" for p in _get_port_names_without_mode(filtered_sax_model)])
        filtered_netlist['instances'][instance_name]['component'] = instance_name
    
    # # TODO: Make it so that I don't have to build a circuit / sanitize the netlist, that takes forever
    # # Honestly, I think that I   
    # # TODO: Remove unnecessary ports from netlist by terminating them.
    # for external_port_name, internal_instance_port in sanitized_netlist["ports"].items():
    #     instance_name, internal_port_name = internal_instance_port.split(",")
    #     model_name = sanitized_netlist['instances'][instance_name]['component']
    #     filtered_sax_model = filtered_sax_models[model_name]
    #     pass

    ports = filtered_netlist['ports']
    ports_to_remove = []
    for ext_port, port_designator in ports.items():
        if not port_designator in filtered_port_designators:
            ports_to_remove.append(ext_port)
    for port in ports_to_remove:
        ports.pop(port)



    new_separator = generate_valid_separator(netlist["instances"].keys())
    sanitized_netlist = sanitize_instance_names(filtered_netlist, old_separator="~", new_separator=new_separator)

    _dummy_sax_model, _ = sax.circuit(sanitized_netlist, filtered_sax_models)
    dummy_sax_model = _get_filtered_sax_model(_dummy_sax_model, external_port_directionality,simulation_parameters.mode_identifiers)
    filter_bank_input_port_modes, filter_bank_output_port_modes = _get_port_mode_luts(dummy_sax_model)
    # dummy_sax_model()
    # filtered_sax_model = _get_filtered_sax_model(sax_model, port_directionality, simulation_parameters.mode_identifiers)
    _netlist, _models = _block_mode_netlist_and_models(filtered_netlist, filtered_sax_models, external_port_directionality, simulation_parameters.mode_identifiers, filter_bank_input_port_modes, filter_bank_output_port_modes)

    # input_port_modes, output_port_modes = _get_port_mode_luts

    # print(f"Building Model For {sax_model}")

    settings = {}
    for instance_name, _settings in original_settings.items():
        vector_fitting_parameters = _settings["vector_fitting_parameters"]
        sax_settings = _settings["sax_settings"]
        model_name = netlist['instances'][instance_name]['component']        
        # sax_model = filtered_sax_models[model_name]
        sax_model = filtered_sax_models[instance_name]
                
        A, B, C, D = _calculate_state_space_coefficients_from_sax_model(sax_model, sax_settings, vector_fitting_parameters, simulation_parameters)
        f_b = speed_of_light / vector_fitting_parameters["center_wavelength"]
        f_s = 1 / simulation_parameters.dt

        settings.update({_state_space_instance_name(instance_name):{"A":A, "B":B, "C":C, "D":D, "baseband_frequency": f_b, "sampling_frequency": f_s}})

    ## TODO: Fill in empty settings..s
    common_mode = simulation_parameters.mode_identifiers[0]
    
    settings.update({_mode_converter_instance_name(port, mode, INPUT_SUFFIX):{"input_mode":mode,"output_mode": common_mode} for port, modes in filter_bank_input_port_modes.items() for mode in modes})
    settings.update({_mode_converter_instance_name(port, mode, OUTPUT_SUFFIX):{"input_mode":common_mode, "output_mode": mode} for port, modes in filter_bank_output_port_modes.items() for mode in modes})
    settings.update({_demultiplexer_instance_name(port):{} for port in filter_bank_input_port_modes.keys()})
    settings.update({_multiplexer_instance_name(port):{} for port in filter_bank_output_port_modes.keys()})
    
    # instantiated_flat_netlist = _instantiate_netlist(netlist, models, settings)

    return _netlist, _models, settings

def _block_mode_netlist_and_models(
    netlist,
    filtered_sax_models,
    port_directionality,
    default_modes,
    filter_bank_input_port_modes, 
    filter_bank_output_port_modes
):
    """
    Returns a dicts defining the instances, connections, and ports of the subcircuit
    as well as a dict of uninstantiated models
    """
    old_netlist = netlist
    old_connections = netlist['connections']
    old_instances = netlist['instances']
    old_ports = netlist['ports']

    netlist = {}
    connections = {}
    instances = {}
    ports = {}
    models = {}

    # The Plan: 
    # 1) Make a dummy sax component out of the netlist 
    # 2) Use dummy sax component to make the Mode demux and mux architecture
    # 3) replace dummy sax component with the filter bank

    for instance_name, instance_data in old_netlist['instances'].items():
        model_name = instance_data['component']
        sax_model = filtered_sax_models[model_name]
        input_port_modes, output_port_modes = _get_port_mode_luts(sax_model)
        state_space_input_port_names, state_space_output_port_names = _state_space_port_names(input_port_modes, output_port_modes)
        DiscreteStateSpace = discrete_state_space(
            len(state_space_input_port_names), 
            len(state_space_output_port_names),
            input_port_names=state_space_input_port_names,
            output_port_names=state_space_output_port_names,
        )
        state_space_instance_name = _state_space_instance_name(instance_name)
        state_space_model_name = _state_space_model_name(model_name)
        models[state_space_model_name] = DiscreteStateSpace
        instances[state_space_instance_name] = state_space_model_name

    
    mode_demultiplexers = {}
    # filter_bank_input_port_names = []
    for input_port, modes in filter_bank_input_port_modes.items():        
        mode_demultiplexers[input_port] = mode_demultiplexer(modes, input_port_name=DEMULTIPLEXER_IN_PORT_NAME,output_port_suffix=DEMULTIPLEXER_OUT_PORT_SUFFIX)
        # filter_bank_input_port_names += [_state_space_port_name(input_port, mode) for mode in modes]

    mode_multiplexers = {}
    # filter_bank_output_port_names = []
    for output_port, modes in filter_bank_output_port_modes.items():
        mode_multiplexers[output_port] = mode_multiplexer(modes, output_port_name=MULTIPLEXER_OUT_PORT_NAME, input_port_suffix=MULTIPLEXER_IN_PORT_SUFFIX)
        # filter_bank_output_port_names += [_state_space_port_name(output_port, mode) for mode in modes]

    # CHECKPOINT
    # TODO: Finish from this point on 
    models[MODE_CONVERTER_MODEL_NAME] = ModeConverter

    # Add connections between filters in the filter bank
    for src, dst in old_connections.items():
        src_instance_name, src_port_name = src.split(",")
        dst_instance_name, dst_port_name = dst.split(",")
        
        src_state_space_instance_name = _state_space_instance_name(src_instance_name)
        dst_state_space_instance_name = _state_space_instance_name(dst_instance_name)
        for mode in default_modes:
            src_state_space_port_name = _state_space_port_name(src_port_name, mode)
            dst_state_space_port_name = _state_space_port_name(dst_port_name, mode)
            connections[src_state_space_instance_name + ',' + src_state_space_port_name] = dst_state_space_instance_name + ',' + dst_state_space_port_name

    # for ext_port_name, modes in filter_bank_input_port_modes.items():
    #     pass

    # Input Side Demultiplexers and Mode Converters
    for port, demux in mode_demultiplexers.items():
        demux_model_name = _demultiplexer_model_name(port)
        models[demux_model_name] = demux
        demux_instance_name = _demultiplexer_instance_name(port)
        instances[demux_instance_name] = demux_model_name

        modes = filter_bank_input_port_modes[port]
        for mode in modes:
            mode_converter_instance_name = _mode_converter_instance_name(port, mode, INPUT_SUFFIX)
            instances[mode_converter_instance_name] = MODE_CONVERTER_MODEL_NAME
            demux_output = demux_instance_name + "," + mode + DEMULTIPLEXER_OUT_PORT_SUFFIX
            converter_input = mode_converter_instance_name + ',' + 'in'
            converter_output = mode_converter_instance_name + ',' + 'out'
            
            sax_instance_name, sax_port_name = old_ports[port].split(",")
            state_space_input = _state_space_instance_name(sax_instance_name) + ',' + _state_space_port_name(sax_port_name, mode)
            connections[demux_output] = converter_input 
            connections[converter_output] = state_space_input# This line for one of the converters is connecting an output to a state space output
        
        ports[port] = demux_instance_name + "," + DEMULTIPLEXER_IN_PORT_NAME
    
    # Output Side Multiplexers and Mode Converters
    for port, mux in mode_multiplexers.items():
        mux_model_name = _multiplexer_model_name(port)
        models[mux_model_name] = mux
        mux_instance_name = _multiplexer_instance_name(port)
        instances[mux_instance_name] = mux_model_name

        modes = filter_bank_output_port_modes[port]
        for mode in modes:
            mode_converter_instance_name = _mode_converter_instance_name(port, mode, OUTPUT_SUFFIX)
            instances[mode_converter_instance_name] = MODE_CONVERTER_MODEL_NAME
            mux_input = mux_instance_name + "," + mode + MULTIPLEXER_IN_PORT_SUFFIX
            converter_input = mode_converter_instance_name + ',' + 'in'
            converter_output = mode_converter_instance_name + ',' + 'out'
            
            sax_instance_name, sax_port_name = old_ports[port].split(",")
            state_space_output = _state_space_instance_name(sax_instance_name) + ',' + _state_space_port_name(sax_port_name, mode)
            
            connections[state_space_output] = converter_input
            connections[converter_output] = mux_input

        ports[port] = mux_instance_name + "," + MULTIPLEXER_OUT_PORT_NAME

    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": ports,
    }

    
    add_settings_to_netlist(netlist)
    

    # # TODO: Remove the following lines for testing
    # import gravis as gv
    # graph = netlist_to_graph(netlist, models)
    # gv.d3(graph).display()
    
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

def _state_space_model_name(
    sax_model_name,    
):
    return sax_model_name + "_" + STATE_SPACE_MODEL_NAME_BASE

def _state_space_instance_name(
    sax_instance_name,    
):
    return sax_instance_name + "_" + STATE_SPACE_INSTANCE_NAME_BASE

def _state_space_port_name(
    port,
    mode,
):
    return port + "_" + mode

def _state_space_port_names(
    input_port_modes,
    output_port_modes,
):
    input_port_names = []
    for input_port, modes in input_port_modes.items():
        input_port_names += [_state_space_port_name(input_port, mode) for mode in modes]

    output_port_names = []
    for output_port, modes in output_port_modes.items():
        output_port_names += [_state_space_port_name(output_port, mode) for mode in modes]
    return input_port_names, output_port_names

# def _get_filtered_sax_model(
#     sax_model: sax.Model,
#     port_directionality,
#     default_modes,
# ):
#     input_ports_to_remove = {port_name for port_name, direction in port_directionality.items() if direction=='output'}
#     output_ports_to_remove = {port_name for port_name, direction in port_directionality.items() if direction=='input'}

#     def filtered_sax_model(**kwargs):
#         """
#         The port_directionality field allows us to ignore data
#         in the sdict and select only the relationships necessary
#         for the specified directionality
#         """
#         sdict = sax_model(**kwargs)
#         # print(sdict.keys())
#         # print()
#         sdict = sax.multimode(sdict, modes=default_modes)
#         # print(sdict.keys())
#         # print()
#         # print()

#         def is_allowed(key):
#             dst, src = key
#             src_port, _ = src.split("@")
#             dst_port, _ = dst.split("@")
#             src_valid = not src_port in input_ports_to_remove
#             dst_valid = not dst_port in output_ports_to_remove
#             return src_valid and dst_valid

#         sdict = {k:v for k, v in sdict.items() if is_allowed(k)}

#         return sdict

#     return filtered_sax_model

import inspect
import functools

def _get_filtered_sax_model(
    sax_model: sax.Model,
    port_directionality,
    default_modes,
):
    input_ports_to_remove = {
        port_name
        for port_name, direction in port_directionality.items()
        if direction == 'output'
    }

    output_ports_to_remove = {
        port_name
        for port_name, direction in port_directionality.items()
        if direction == 'input'
    }

    @functools.wraps(sax_model)
    def filtered_sax_model(*args, **kwargs):
        """
        The port_directionality field allows us to ignore data
        in the sdict and select only the relationships necessary
        for the specified directionality
        """
        sdict = sax_model(*args, **kwargs)

        sdict = sax.multimode(sdict, modes=default_modes)

        def is_allowed(key):
            dst, src = key
            src_port, _ = src.split("@")
            dst_port, _ = dst.split("@")

            src_valid = src_port not in input_ports_to_remove
            dst_valid = dst_port not in output_ports_to_remove

            return src_valid and dst_valid

        return {k: v for k, v in sdict.items() if is_allowed(k)}

    # preserve original signature
    filtered_sax_model.__signature__ = inspect.signature(sax_model)

    return filtered_sax_model

def _normalize_mode_name(mode) -> str:
    return str(mode).lower()

def _ordered_from_set(mode_set, mode_identifiers=None):
    if mode_identifiers is None:
        return tuple(sorted(mode_set, key=_normalize_mode_name))
 
    wanted = [_normalize_mode_name(m) for m in mode_identifiers]
    present = {_normalize_mode_name(m): m for m in mode_set}
 
    ordered = [present[m] for m in wanted if m in present]
    extras = [m for m in mode_set if _normalize_mode_name(m) not in set(wanted)]
    ordered.extend(sorted(extras, key=_normalize_mode_name))
    return tuple(ordered)
 
 
def _get_port_mode_luts(
    sax_model: sax.Model,
    mode_identifiers=None,
):
    input_port_modes = {}
    output_port_modes = {}
 
    for o, i in sax_model().keys():
        in_port, in_mode = i.split("@")
        out_port, out_mode = o.split("@")
        input_port_modes.setdefault(in_port, set()).add(in_mode)
        output_port_modes.setdefault(out_port, set()).add(out_mode)
 
    input_port_modes = {
        port: _ordered_from_set(modes, mode_identifiers)
        for port, modes in input_port_modes.items()
    }
    output_port_modes = {
        port: _ordered_from_set(modes, mode_identifiers)
        for port, modes in output_port_modes.items()
    }
 
    return input_port_modes, output_port_modes
 


# def _get_port_mode_luts(
#     sax_model: sax.Model,
# ):
#     input_port_modes = {}
#     output_port_modes = {}
#     for o, i in sax_model().keys():
#         in_port, in_mode = i.split('@')
#         out_port, out_mode = o.split('@')
#         input_port_modes.setdefault(in_port, set()).add(in_mode)
#         output_port_modes.setdefault(out_port, set()).add(out_mode)

#     return input_port_modes, output_port_modes

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

def _calculate_state_space_coefficients_from_sax_model(sax_model, sax_settings, vector_fitting_parameters, simulation_parameters):
    input_port_modes, output_port_modes = _get_port_mode_luts(sax_model, simulation_parameters.mode_identifiers)
    sax_model_signature = inspect.signature(sax_model).parameters

    # Some sax models are constant over wavelength. This accounts for those.
    if not "wl" in sax_model_signature:
        sdict = sax_model(**sax_settings)
        input_ports = [f"{port}@{mode}" for port, modes in input_port_modes.items() for mode in modes]
        output_ports = [f"{port}@{mode}" for port, modes in output_port_modes.items() for mode in modes]
        S = dict_to_rect_matrix(sdict, input_ports=input_ports, output_ports=output_ports)
        
        # Order r = 1 model
        r = 1
        m = len(input_ports)
        M = r*m
        q = len(output_ports)
        A = jnp.zeros((M, M), dtype=complex)
        B = jnp.zeros((M, m), dtype=complex)
        C = jnp.zeros((q, M), dtype=complex)
        D = S[0, :, :]

        return A, B, C, D
    elif False:
        # TODO: Account for the case where the function is constant over wavelength, by wl happens to be a parameter
        # Perhaps the best way to account for that is to put a check in the z_domain code
        pass

    f_min = speed_of_light / max(vector_fitting_parameters['spectral_range'])
    f_max = speed_of_light / min(vector_fitting_parameters['spectral_range'])
    f_center = speed_of_light / vector_fitting_parameters['center_wavelength']
    # f_center = 192.9e12
    frequency = jnp.linspace(f_min, f_max, vector_fitting_parameters["num_frequency_samples"])
    sdict = sax_model(wl=1e6*speed_of_light/frequency, **sax_settings)
    input_ports = [f"{port}@{mode}" for port, modes in input_port_modes.items() for mode in modes]
    output_ports = [f"{port}@{mode}" for port, modes in output_port_modes.items() for mode in modes]
    s_params = dict_to_rect_matrix(sdict, input_ports=input_ports, output_ports=output_ports)
    min_order = vector_fitting_parameters["min_model_order"]
    max_order = vector_fitting_parameters["max_model_order"]
    # Checkpoint
    sampling_frequency = 1/simulation_parameters.dt
    
    ### TODO: REMOVE THIS LINE USED FOR TESTING
    # vector_fitting_parameters["model_order"] = 20
    
    if vector_fitting_parameters["model_order"] is None:
        poles, residues, feedthrough, mean_squared_error = optimize_order_vector_fitting_discrete(min_order, max_order, s_params, frequency, f_center, sampling_frequency, sign_convention=PHYSICIST)
    else:
        poles, residues, feedthrough, mean_squared_error = vector_fitting_discrete(vector_fitting_parameters["model_order"], s_params, frequency, f_center, sampling_frequency, sign_convention=PHYSICIST)

    A, B, C, D = state_space_discrete(poles, residues, feedthrough)

    return A, B, C, D

class SParameterPlaceholder(Placeholder):
    """
    Using the Component Factory Below
    """

def optical_s_parameter_placeholder(
    sax_model: sax.Model, 
    port_directionality: dict = None,
    default_modes: list|tuple|str = DEFAULT_MODES,
)-> type[SParameterPlaceholder]:
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
    
    # default_port_directionality = port_directionality


    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)
    
    BaseSParameterSax = SParameterPlaceholder # Freeze the reference
    class SpecificSParameterPlaceholder(BaseSParameterSax):
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
            # Rename kwargs to be more accurate
            **kwargs,
            # sax_settings: dict = None,
            # vector_fitting_parameters = None,
            # delay_compensation: int = 0,
            # port_directionality = {},
        ):
            """
            TODO: Add documentation for each of the kwargs
            """
            # default_vector_fitting_parameters = {
            #     "model_order": None,
            #     "min_model_order": 2,
            #     "max_model_order": 100,
            #     "num_frequency_samples": 1000,
            #     "center_wavelength": 1.55e-6,
            #     "spectral_range": (1.5e-6, 1.6e-6),
            #     # NOTE: Currently, a user CAN change the spectral range parameter (the default is set by )
            # }
            self.sax_model = sax_model
            self.settings = kwargs
            self.settings.setdefault('sax_settings', {})
            self.settings.setdefault('group_id', None)
            self.settings.setdefault('vector_fitting_parameters', _default_vector_fitting_parameters)
            self.settings.setdefault('delay_compensation', 0)
            self.settings.setdefault('port_directionality', {})
            # self.sax_model = sax_model
            # self.sax_settings = self.settings.setdefault('sax_settings', {})
            # self.vector_fitting_parameters = self.settings.setdefault('vector_fitting_parameters', default_vector_fitting_parameters)
            # self.delay_compensation = self.settings.setdefault('delay_compensation', 0)
            # self.port_directionality = self.settings.setdefault('port_directionality', {})
    
    return SpecificSParameterPlaceholder