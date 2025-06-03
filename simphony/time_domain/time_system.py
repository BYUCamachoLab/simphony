from abc import ABC, abstractmethod
from jax.typing import ArrayLike
import numpy as np
from simphony.time_domain.pole_residue_model import PoleResidueModel
import jax.numpy as jnp
from simphony.simulation import SimDevice

class TimeSystem(ABC):
    def __init__(self) -> None:
        pass

class BlockModeSystem(TimeSystem):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, input_signal) -> ArrayLike:
        """Compute the system response."""
        raise NotImplementedError

class SampleModeSystem(TimeSystem):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def init_state(self):
        """Initialize the state of the system."""
        raise NotImplementedError

    @abstractmethod
    def step(self, x_prev: jnp.ndarray, *inputs_tuple) -> jnp.ndarray:
        """Compute the next state of the system."""
        raise NotImplementedError
    




# def my_dlsim(system, u, t=None, x0=None):
#         out_samples = len(u)
#         stoptime = (out_samples - 1) * system.dt

#         xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
#         yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
#         tout = np.linspace(0.0, stoptime, num=out_samples)

#         xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)
#         if x0 is not None:
#             xout[0, :] = x0

#         u_dt = u

#         # Simulate the system
#         for i in range(0, out_samples - 1):
#             xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
#                             np.dot(system.B, u_dt[i, :]))
#             yout[i, :] = (np.dot(system.C, xout[i, :]) +
#                         np.dot(system.D, u_dt[i, :]))

#         # Last point
#         yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
#                                 np.dot(system.D, u_dt[out_samples-1, :]))

#         return tout, yout, xout

def my_dlsim(system, u, t=None, x0=None):
    out_samples = len(u)
    stoptime = (out_samples - 1) * system.dt

    xout = jnp.zeros((out_samples, system.A.shape[0]), dtype=jnp.complex64)
    yout = jnp.zeros((out_samples, system.C.shape[0]), dtype=jnp.complex64)
    tout = jnp.linspace(0.0, stoptime, num=out_samples)

    # Manually create a mutable copy because JAX arrays are immutable
    xout = xout.at[0, :].set(jnp.zeros((system.A.shape[1],), dtype=jnp.complex64))
    if x0 is not None:
        xout = xout.at[0, :].set(x0.flatten())
        # xout = xout.at[0, :].set(x0)

    u_dt = u

    # Simulate the system (using Python loop since JAX loops need special handling)
    xout_list = [xout[0, :]]
    yout_list = []

    for i in range(out_samples - 1):
        x_next = jnp.dot(system.A, xout_list[-1]) + jnp.dot(system.B, u_dt[i, :])
        y_curr = jnp.dot(system.C, xout_list[-1]) + jnp.dot(system.D, u_dt[i, :])
        xout_list.append(x_next)
        yout_list.append(y_curr)

    # Final output
    y_last = jnp.dot(system.C, xout_list[-1]) + jnp.dot(system.D, u_dt[-1, :])
    yout_list.append(y_last)

    # Stack results
    xout = jnp.stack(xout_list)
    yout = jnp.stack(yout_list)

    return tout, yout, xout

def my_dlsimworks(system, u, t=None, x0=None):
        out_samples = len(u)
        stoptime = (out_samples) * system.dt

        # xout = np.zeros((out_samples, system.A.shape[0]), dtype=complex)
        # yout = np.zeros((out_samples, system.C.shape[0]), dtype=complex)
        # tout = np.linspace(0.0, stoptime, num=out_samples)
        xout = jnp.zeros((out_samples, system.A.shape[0]), dtype=jnp.complex128)
        yout = jnp.zeros((out_samples, system.C.shape[0]), dtype=jnp.complex128)
        tout = jnp.linspace(0.0, stoptime, num=out_samples)


        # xout[0, :] = np.zeros((system.A.shape[1],), dtype=complex)
        # if x0 is not None:
        #     xout[0, :] = x0
        xout = xout.at[0, :].set(jnp.zeros((system.A.shape[1],), dtype=jnp.complex128))
        if x0 is not None:
            xout = xout.at[0, :].set(x0.flatten())
            # xout = xout.at[0, :].set(x0)
            

        u_dt = u

        # # Simulate the system
        # for i in range(0, out_samples):
        #     xout[i, :] = (np.dot(system.A, xout[i, :]) +
        #                     np.dot(system.B, u_dt[i, :]))
        #     yout[i, :] = (np.dot(system.C, xout[i, :]) +
        #                 np.dot(system.D, u_dt[i, :]))

        # # Last point
        # yout[out_samples-1, :] = (np.dot(system.C, xout[out_samples-1, :]) +
        #                         np.dot(system.D, u_dt[out_samples-1, :]))
        # Simulate the system
        for i in range(0, out_samples):
            xout = xout.at[i, :].set(jnp.dot(system.A, xout[i, :]) +
                                    jnp.dot(system.B, u_dt[i, :]))
            yout = yout.at[i, :].set(jnp.dot(system.C, xout[i, :]) +
                                    jnp.dot(system.D, u_dt[i, :]))

        # Last point
        yout = yout.at[out_samples - 1, :].set(
            jnp.dot(system.C, xout[out_samples - 1, :]) +
            jnp.dot(system.D, u_dt[out_samples - 1, :])
        )

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



# class TimeSystemIIR(TimeSystem):
#     def __init__(self, pole_model: PoleResidueModel, ports= None ) -> None:
#         super().__init__()
#         self.sys = pole_model.generate_sys_discrete()
#         self.num_ports = self.sys.B.shape[1] 
#         if ports is None:
#             self.ports = [f'o{i}' for i in range(self.num_ports)]
#         else:
#             self.ports = ports
        
#         self.state_vector = None
        

#     def response(self, inputs: dict, time_sim = True) -> ArrayLike:
#         # if state_vector is not None:
#         #      self.state_vector = state_vector

#         first_key = next(iter(inputs))
#         N = inputs[first_key].shape
#         responses = {}
        
#         input = jnp.hstack([value.reshape(-1, 1) 
#                             for value in inputs.values()])
#         if not time_sim:
#             t,y_out,x_out = my_dlsim(self.sys, input, x0 = self.state_vector)
#         else:
#             t,y_out,x_out = my_dlsimworks(self.sys, input, x0 = self.state_vector)

#         self.state_vector = x_out

#         j = 0
#         for i in self.ports:
#              responses[i] = y_out[:,j]
#              j += 1

#         return responses,t
    
#     def reset(self):
#         self.state_vector = None

class TimeSystemIIR(TimeSystem):
    def __init__(self, pole_model: PoleResidueModel, ports=None):
        super().__init__()
        # Generate the discrete‐time state‐space (A,B,C,D) from your pole‐residue model:
        #   self.sys.A, self.sys.B, self.sys.C, self.sys.D
        self.sys = pole_model.generate_sys_discrete()

        # Number of “input” ports is B.shape[1], number of “output” ports is C.shape[0]
        self.num_in_ports  = self.sys.B.shape[1]
        self.num_out_ports = self.sys.C.shape[0]

        if ports is None:
            # default port names: o0, o1, ...  (one per output channel)
            self.ports = [f'o{i}' for i in range(self.num_out_ports)]
        else:
            self.ports = ports

    def init_state(self):
        """
        Return the initial x(0) for this IIR system. 
        If you want x(0)=0, make a zeros vector of shape (A.shape[0],).
        """
        n_states = self.sys.A.shape[0]
        return jnp.zeros((n_states,), dtype=jnp.complex128)

    def step(self, x_prev: jnp.ndarray, *inputs_tuple):
        """
        A pure function (no in-place mutation). 
        If there are N inputs to this IIR block, `inputs_tuple` is a tuple of length N,
        so we first stack them into a 1-D vector `u_row`.
        Then:
           x_next = A @ x_prev + B @ u_row
           y_row  = C @ x_prev + D @ u_row
           return (x_next, tuple(y_row_i for each scalar output)).
        """
        # Stack all inputs into one (n_inputs,) vector:
        u_row = jnp.stack(inputs_tuple, axis=0)    # shape = (n_inputs,)
        A, B, C, D = self.sys.A, self.sys.B, self.sys.C, self.sys.D
        x_next = A @ x_prev + B @ u_row            # shape = (n_states,)
        y_full = C @ x_prev + D @ u_row            # shape = (n_outputs,)
        # We need to return a Python tuple of length = n_outputs:
        #   e.g. (y_full[0], y_full[1], …)
        return x_next, tuple(jnp.atleast_1d(y_full[i]) for i in range(y_full.shape[0]))

    def response(self, inputs: dict, time_sim=True) -> ArrayLike:
        # if state_vector is not None:
        #      self.state_vector = state_vector

        first_key = next(iter(inputs))
        N = inputs[first_key].shape
        responses = {}

        input = jnp.hstack([value.reshape(-1, 1)
                                for value in inputs.values()])
        if not time_sim:
            t,y_out,x_out = my_dlsim(self.sys, input, x0 = self.state_vector)
        else:
            t,y_out,x_out = my_dlsimworks(self.sys, input, x0 = self.state_vector)

        self.state_vector = x_out

        j = 0
        for i in self.ports:
            responses[i] = y_out[:,j]
            j += 1

        return responses,t