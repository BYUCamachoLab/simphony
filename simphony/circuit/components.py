import inspect
from typing import Tuple

# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
# import gravis as gv
import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel
from simphony.time_domain.pole_residue_model import IIRModelBaseband

from simphony.utils import dict_to_matrix
from copy import deepcopy
from functools import partial

# from simphony.utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph
# from copy import deepcopy
from simphony.signals import    steady_state_optical_signal, \
                                sample_mode_electrical_signal, \
                                sample_mode_optical_signal, \
                                sample_mode_logic_signal, \
                                block_mode_optical_signal, \
                                block_mode_electrical_signal, \
                                block_mode_logic_signal, \
                                complete_steady_state_inputs, \
                                complete_sample_mode_inputs

import jax
import jax.numpy as jnp

from simphony.utils import dict_to_matrix

class Signal: ## TODO: Make an actual base class
    ...
class Component:
    electrical_ports = []
    logic_ports = []
    optical_ports = []

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
            spectral_range=(1.5,1.6),
            **sax_settings
        ):
            # super().__init__(**settings)
            self.settings = sax_settings
            self.spectral_range = spectral_range

        def initial_state(self):
            self.state_space_models = []
            wvl_microns = 1e6*jnp.linspace(self.spectral_range[0], self.spectral_range[1], 1000)
            s_params = self.s_parameters(wl=wvl_microns)
            
            for optical_wl in self.optical_wls:
                # TODO: Make IIRModelBaseband take a sax.SDict
                iir_model = IIRModelBaseband(
                    wvl_microns=wvl_microns,
                    center_wvl=optical_wl, 
                    s_params=dict_to_matrix(s_params), 
                    sampling_period=self.sampling_period, 
                    order=50
                )

                # Convert pole-resiude model to state space model
                self.state_space_models.append(iir_model.generate_sys_discrete())
            
            return jnp.zeros((self.state_space_models[0].A.shape[0],), dtype=complex)
            
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

        def step(
            self,
            inputs: dict,
            state: jax.Array,
        ):
            return inputs, state

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
            complete_steady_state_inputs(inputs)
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
                outputs[port] = steady_state_optical_signal(
                                    field=output_matrix[:, i],
                                    wl=wls,
                                    polarization=inputs[port].polarization
                                )
            
        
            
            return outputs

    return SParameterSax


class BlockModeComponent(Component):
    def __init__(
        self
        # , optical_ports=None, electrical_ports=None, logic_ports=None
    ) -> None:
        ...
        # super().__init__(optical_ports, electrical_ports, logic_ports)

    # IDK the best name for this method! Maybe run, but that is confusing
    def response(self, input_signal: ArrayLike) -> ArrayLike:
        """Compute the system response."""
        raise NotImplementedError


class SampleModeComponent(Component):
    delay_compensation = 0 # Applies to all port-to-port relationships
    # simulation_parameters={}
    def set_sample_mode_simulation_parameters(
            self,
            optical_wls, 
            electrical_wls,
            sampling_period,
            num_time_steps, 
        ):
        # self.simulation_parameters['sampling_period'] = sampling_period
        # self.simulation_parameters['num_time_steps'] = num_time_steps
        # self.simulation_parameters['delay_compensation'] = delay_compensation

        self.sampling_period = sampling_period
        self.sampling_rate = 1 / sampling_period
        self.num_time_steps = num_time_steps
        self.optical_wls = optical_wls
        self.electrical_wls = electrical_wls
    
    def _sample_mode_restart(
        self,
        optical_wls, 
        electrical_wls,
        sampling_period,
        num_time_steps,
        max_delay_compensation,
    ):
        self.set_sample_mode_simulation_parameters(
            optical_wls, 
            electrical_wls,
            sampling_period,
            num_time_steps, 
        )
        T = max_delay_compensation - self.delay_compensation + 1
        num_optical_wls = len(optical_wls)
        num_electrical_wls = len(electrical_wls)
        buffered_outputs = {}
        for electrical_port in self.electrical_ports:
            buffered_outputs[electrical_port] = block_mode_electrical_signal(
                field = jnp.zeros((T, num_electrical_wls), dtype=complex),
                wl = electrical_wls,
            )
        for logic_port in self.logic_ports:
            buffered_outputs[logic_port] = block_mode_logic_signal(
                value = jnp.zeros((T, ), dtype=int),
            )
        for optical_port in self.optical_ports:
            buffered_outputs[optical_port] = block_mode_optical_signal(
                field = jnp.zeros((T, num_optical_wls), dtype=complex),
                wl = optical_wls,
                # Use default polarization
            )
        
        self._initial_buffered_outputs = buffered_outputs

    def initial_state(self):
        """
        Returns the initial the state of the system.
        Called by the sample mode simulator after `set_sample_mode_simulation_parameters`
        """
        raise NotImplementedError

    def _initial_state(self):
        return 0, self._initial_buffered_outputs, self.initial_state()
    
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

    def step(self, inputs: dict,  state: jax.Array, ) -> Tuple[jax.Array, dict[str, Signal]]:
        """Compute the next state of the system."""
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=(0,))
    def _step(self, inputs: dict, state: jax.Array):
        time_step = state[0]
        buffered_outputs = state[1]
        internal_state = state[2]
        outputs, output_state = self.step(inputs, internal_state)
        
        return outputs, (time_step+1, buffered_outputs, output_state)


    