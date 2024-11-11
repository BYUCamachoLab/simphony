from simphony.time_domain.time_system import TimeSystem
from simphony.time_domain.pole_residue_model import PoleResidueModel
from jax.typing import ArrayLike
import jax.numpy as jnp

# def pole_residue_to_time_system(pole_residue_model: PoleResidueModel) -> TimeSystem:
#     pass


def gaussian_pulse(t, t0, std, a=1.0) -> ArrayLike:
    return a * jnp.exp(-(t - t0)**2 / std**2)