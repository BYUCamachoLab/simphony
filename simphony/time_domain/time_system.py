from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import numpy as np
from scipy.signal import  StateSpace, lsim,dlsim
from simphony.time_domain.pole_residue_model import PoleResidueModel
import jax.numpy as jnp


class TimeSystem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        pass

class CVF_Baseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel, t) -> None:
        self.A,self.B,self.C,self.D = pole_model.compute_state_space_model()
        self.num_ports = pole_model.num_ports
        self.t = t
        super().__init__()

    def response(self, inputs: dict) -> ArrayLike:
        N = inputs['o0'].shape[0]
        responses = {
            'o0': jnp.zeros((N), dtype=complex)
            }
        for i in range(0,self.num_ports):
            responses[f'o{i}'] = jnp.zeros((N), dtype=complex)
        
        sys = StateSpace(self.A,self.B,self.C,self.D)

        t_out,y_out,_ = lsim(sys, input, self.t)
        
        for i in range(0,self.num_ports):
             responses[f'o{i}'] = y_out[i]

        return t_out, y_out

class IIRModelBaseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel, t) -> None:
        self.sys = pole_model.generate_sys_discrete()
        self.t = t
        super().__init__()

    def response(self, inputs: ArrayLike) -> ArrayLike:
        
        t_out,y_out,_ = dlsim(self.sys, inputs, self.t)
        

        return t_out, y_out