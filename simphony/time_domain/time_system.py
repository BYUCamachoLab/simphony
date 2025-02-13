from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import numpy as np
from simphony.time_domain.pole_residue_model import PoleResidueModel
import jax.numpy as jnp
from simphony.simulation import SimDevice


class TimeSystem(SimDevice):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def response(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        pass
    
    @abstractmethod
    def reset(self):
        pass




def my_dlsim(system, u, t=None, x0=None):
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt

        xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
        yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
        tout = np.linspace(0.0, stoptime, num=out_samples)

        xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)
        if x0 is not None:
            xout[0, :] = x0

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

def my_dlsimworks(system, u, t=None, x0=None):
        out_samples = len(u)
        stoptime = (out_samples) * system.dt

        xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
        yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
        tout = np.linspace(0.0, stoptime, num=out_samples)

        xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)

        if x0 is not None:
            xout[0, :] = x0

        u_dt = u

        # Simulate the system
        for i in range(0, out_samples):
            xout[i, :] = (np.dot(system.A, xout[i, :]) +
                            np.dot(system.B, u_dt[i, :]))
            yout[i, :] = (np.dot(system.C, xout[i, :]) +
                        np.dot(system.D, u_dt[i, :]))

        # Last point
        yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
                                np.dot(system.D, u_dt[out_samples-1, :]))

        return tout, yout, xout



class TimeSystemIIR(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel, ports= None ) -> None:
        super().__init__()
        self.sys = pole_model.generate_sys_discrete()
        self.num_ports = self.sys.B.shape[1] 
        if ports is None:
            self.ports = [f'o{i}' for i in range(self.num_ports)]
        else:
            self.ports = ports
        
        self.state_vector = None
        

    def response(self, inputs: dict) -> ArrayLike:
        # if state_vector is not None:
        #      self.state_vector = state_vector

        N = inputs['o0'].shape
        responses = {}
        
        input = jnp.hstack([value.reshape(-1, 1) 
                            for value in inputs.values()])
        t,y_out,x_out = my_dlsimworks(self.sys, input, x0 = self.state_vector)
        self.state_vector = x_out

        j = 0
        for i in self.ports:
             responses[i] = y_out[:,j]
             j += 1

        return responses,t
    
    def reset(self):
        self.state_vector = None
    
    