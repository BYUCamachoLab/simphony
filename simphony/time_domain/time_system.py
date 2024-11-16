from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import numpy as np
from scipy.signal import  StateSpace, lsim,dlsim
from simphony.time_domain.pole_residue_model import PoleResidueModel

class TimeSystem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        pass

class CVF_Baseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel) -> None:
        self.A,self.B,self.C,self.D = pole_model.compute_state_space_model()
        super().__init__()

    def response(self, input: ArrayLike) -> ArrayLike:
        
        N = int(len(input))
        T = 4e-11
        t = np.linspace(0,T,N)

        sys = StateSpace(self.A,self.B,self.C,self.D)

        t_out,y_out,_ = lsim(sys, input, t)
        

        return t_out, y_out

class IIRModelBaseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel) -> None:
        self.sys = pole_model.generate_sys_discrete()
        super().__init__()

    def response(self, input: ArrayLike) -> ArrayLike:
        N = int(len(input))
        T = 4e-11
        t = np.linspace(0,T,N)

        t_out,y_out,_ = dlsim(self.sys, input, t)
        

        return t_out, y_out