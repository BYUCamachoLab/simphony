from simphony.time_domain.time_system import TimeSystem, IIRModelBaseband_to_time_system, CVF_Baseband_to_time_system
from simphony.time_domain.pole_residue_model import PoleResidueModel, IIRModelBaseband, CVFModelBaseband
from jax.typing import ArrayLike
import jax.numpy as jnp

def pole_residue_to_time_system(pole_residue_model: PoleResidueModel) -> TimeSystem:
    if isinstance(pole_residue_model, IIRModelBaseband):
        return IIRModelBaseband_to_time_system(pole_residue_model)
    if isinstance(pole_residue_model, CVFModelBaseband):
        return CVF_Baseband_to_time_system(pole_residue_model)



def gaussian_pulse(t, t0, std, a=1.0) -> ArrayLike:
    return a * jnp.exp(-(t - t0)**2 / std**2)