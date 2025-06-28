import inspect

# from simphony.libraries.analytic.component_types import OpticalComponent, ElectricalComponent, LogicComponent
# import gravis as gv
import sax
from jax.typing import ArrayLike
from sax.saxtypes import Model as SaxModel

# from simphony.utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph
# from copy import deepcopy
from simphony.signals import optical_signal, complete_steady_state_inputs

import jax
import jax.numpy as jnp

from simphony.utils import dict_to_matrix


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

class OpticalSParameterComponent(SteadyStateComponent):
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
    class SParameterSax(OpticalSParameterComponent):
        optical_ports = sax.get_ports(sax_model)
        _num_ports = len(optical_ports)
        
        def __init__(self, **settings):
            # super().__init__(**settings)
            self.settings = settings

        # @staticmethod
        # @jax.jit
        def s_parameters( 
            self,
            wl: ArrayLike,
        ):
            return sax_model(wl, **self.settings)
        
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
                outputs[port] = optical_signal(
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
    def __init__(self) -> None:
        ...
        # super().__init__()

    def initial_state(self):
        """Returns the initial the state of the system."""
        raise NotImplementedError

    def step(self, previous_state, inputs: dict) -> jnp.ndarray:
        """Compute the next state of the system."""
        raise NotImplementedError
    
    def _step(self):
        pass