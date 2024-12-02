from simphony.time_domain.time_system import TimeSystem, IIRModelBaseband_to_time_system
from simphony.time_domain.pole_residue_model import PoleResidueModel, IIRModelBaseband, CVFModelBaseband
from jax.typing import ArrayLike
import jax.numpy as jnp

def pole_residue_to_time_system(pole_residue_model: PoleResidueModel) -> TimeSystem:
    if isinstance(pole_residue_model, IIRModelBaseband):
        return IIRModelBaseband_to_time_system(pole_residue_model)
    # if isinstance(pole_residue_model, CVFModelBaseband):
    #     return CVF_Baseband_to_time_system(pole_residue_model)



def gaussian_pulse(t, t0, std, a=1.0) -> ArrayLike:
    return a * jnp.exp(-(t - t0)**2 / std**2)

def smooth_rectangular_pulse(t, t_start, t_end, width=None):
    """
    A useful function for testing time-domain components

    t = np.linspace(0, 1e-10, 1000)
    pulse = smooth_rectangular_pulse(t, t_start=20e-12, t_end=40e-12)
    plt.plot(t, pulse)
    plt.show()
    """
    if width is None:
        width = (t_end - t_start) / 100.0
    # Transition functions
    rise = 0.5 * (1 + jnp.tanh((t - t_start) / width))
    fall = 0.5 * (1 + jnp.tanh((t_end - t) / width))
    
    # Smooth pulse
    pulse = rise * fall
    return pulse