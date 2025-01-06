from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import numpy as np
from simphony.time_domain.pole_residue_model import PoleResidueModel
import jax.numpy as jnp


class TimeSystem(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        pass



def my_dlsim(system, u, t=None, x0=None):
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt

        xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
        yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
        tout = np.linspace(0.0, stoptime, num=out_samples)

        xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)

        u_dt = u

        # Simulate the system
        for i in range(0, out_samples - 1):
            xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                            np.dot(system.B, u_dt[i, :]))
            yout[i, :] = (np.dot(system.C, xout[i, :]) +
                        np.dot(system.D, u_dt[i, :]))

        # Last point
        yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                                np.dot(system.D, u_dt[out_samples-1, :]))

        return tout, yout, xout



class IIRModelBaseband_to_time_system(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel) -> None:
        self.sys = pole_model.generate_sys_discrete()
        self.num_ports = self.sys.B.shape[1] 
        
        super().__init__()

    def response(self, inputs: dict) -> ArrayLike:
        N = inputs['o0'].shape
        responses = {
            'o0': jnp.zeros((N), dtype=complex)
            }
        for i in range(1, self.num_ports):
            responses[f'o{i}'] = jnp.zeros((N), dtype=complex)
        

        
        input = jnp.hstack([value.reshape(-1, 1) 
                            for value in inputs.values()])
        
        

        t,y_out,_ = my_dlsim(self.sys, input)
        for i in range(self.num_ports):
             responses[f'o{i}'] = y_out[:,i]

        return responses
    
    