from flax import struct
import jax.numpy as jnp
from typing import Union
# from dataclasses import field

@struct.dataclass
class OpticalSignal:
    field: jnp.ndarray
    wl: float
    polarization: jnp.ndarray

@struct.dataclass
class ElectricalSignal:
    field: jnp.ndarray
    wl: float

@struct.dataclass
class LogicSignal:
    voltage: jnp.ndarray

def optical_signal(
    field: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j,
    wl: float = 1550e-9,
    polarization: Union[list, jnp.ndarray] = (1.0 + 0.0j, 0.0 + 0.0j),
) -> OpticalSignal:
    return OpticalSignal(
        field=jnp.asarray(field, dtype=jnp.complex64),
        wl=float(wl),
        polarization=jnp.asarray(polarization, dtype=jnp.complex64)
    )

def electrical_signal(
    voltage: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j,
    wl: float = 0,
) -> ElectricalSignal:
    return ElectricalSignal(
        voltage=jnp.asarray(voltage, dtype=jnp.complex64),
        wl=float(wl)
    )

def logic_signal(
    field: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j
) -> LogicSignal:
    return LogicSignal(
        field=jnp.asarray(field, dtype=jnp.complex64)
    )



# from flax import struct
# import jax.numpy as jnp

# @struct.dataclass
# class OpticalSignal:
#     field: jnp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jnp.array(0.0 + 0.0j))
#     wl: float = 1550e-9
#     polarization: jnp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jnp.array([1.0 + 0.0j, 0.0 + 0.0j]))

# @struct.dataclass
# class ElectricalSignal:
#     field: jnp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jnp.array(0.0 + 0.0j))
#     wl: float = 1550e-9

# @struct.dataclass
# class LogicSignal:
#     field: jnp.ndarray = struct.field(pytree_node=True, default_factory=lambda: jnp.array(0, dtype=int))

# def optical_signal(
#         field=jnp.array(0.0 + 0.0j), 
#         wl=1550e-9, 
#         polarization=jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
#     ) -> OpticalSignal:
#     return OpticalSignal(
#         field=jnp.asarray(field, dtype=jnp.complex64),
#         wl=float(wl),
#         polarization=jnp.asarray(polarization, dtype=jnp.complex64)
#     )

# def electrical_signal(
#         field=jnp.array(0.0 + 0.0j), 
#         wl=1550e-9, 
#         polarization=jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
#     ) -> ElectricalSignal:
#     return ElectricalSignal(
#         field=jnp.asarray(field, dtype=jnp.complex64),
#         wl=float(wl),
#     )

# def logic_signal(
#         field=jnp.array(0.0 + 0.0j), 
#         wl=1550e-9, 
#         polarization=jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
#     ) -> LogicSignal:
#     return LogicSignal(
#         field=jnp.asarray(field, dtype=jnp.complex64),
#     )