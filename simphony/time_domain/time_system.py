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
        T = 2e-11
        t = np.linspace(0,T,N)

        sys = StateSpace(self.A,self.B,self.C,self.D)

        t_out,y_out,_ = lsim(sys, input, t)
        

        return t_out, y_out

class IIRModelBaseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel) -> None:
        self.A,self.B,self.C,self.D = pole_model.compute_state_space_model()
        self.sampling_freq = pole_model.sampling_freq
        super().__init__()

    def response(self, input: ArrayLike) -> ArrayLike:
        N = int(len(input))
        T = 2e-11
        t = np.linspace(0,T,N)

        sys = StateSpace(self.A,self.B,self.C,self.D,dt = 1/self.sampling_freq)

        t_out,y_out,_ = dlsim(sys, input, t)
        

        return t_out, y_out