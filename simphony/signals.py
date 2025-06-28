from flax import struct
import jax.numpy as jnp
from typing import Union


STEADY_STATE = 0 # Superpositions of Discrete Frequencies
TEMPORAL = 1 # Time-varying signal at a carrier frequency



@struct.dataclass
class SteadyStateOpticalSignal:
    # Array of complex amplitudes per wavelength
    field: jnp.ndarray  # shape: (N,) or (N, T), where N is number of wavelengths, T is time samples
    wl: jnp.ndarray     # shape: (N,), corresponding wavelengths
    polarization: jnp.ndarray  # shape: (N, 2) or (2,) if fixed polarization

@struct.dataclass
class SteadyStateElectricalSignal:
    voltage: jnp.ndarray  # shape: (N,) or (N, T)
    wl: jnp.ndarray       # shape: (N,)

@struct.dataclass
class SteadyStateLogicSignal:
    value: jnp.ndarray 


def _steady_state_optical_signal(
    field: Union[float, complex, list, jnp.ndarray] = 0.0 + 0.0j,
    wl: Union[float, list, jnp.ndarray] = 1550e-9,
    polarization: Union[list, jnp.ndarray] = None
) -> SteadyStateOpticalSignal:
    field = jnp.atleast_1d(jnp.asarray(field, dtype=jnp.complex64))
    wl = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))

    # Default polarization
    if polarization is None:
        polarization = jnp.tile(jnp.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64), (wl.shape[0], 1))
    else:
        polarization = jnp.asarray(polarization, dtype=jnp.complex64)

    return SteadyStateOpticalSignal(field=field, wl=wl, polarization=polarization)

def optical_signal(mode=STEADY_STATE,**kwargs):
    if mode==STEADY_STATE:
        return _steady_state_optical_signal(**kwargs)
    elif mode==TEMPORAL:
        return None



def _steady_state_electrical_signal(
    voltage: Union[float, complex, list, jnp.ndarray] = [0.0 + 0.0j],
    wl: Union[float, list, jnp.ndarray] = [0],
) -> SteadyStateOpticalSignal:
    voltage = jnp.asarray(voltage, dtype=jnp.complex64)
    wl = jnp.asarray(wl, dtype=jnp.float32)
    return SteadyStateElectricalSignal(voltage=voltage, wl=wl)

def electrical_signal(mode=STEADY_STATE, **kwargs):
    if mode==STEADY_STATE:
        return _steady_state_electrical_signal(**kwargs)
    elif mode==TEMPORAL:
        return None

def _steady_state_logic_signal(
    value: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j
) -> SteadyStateLogicSignal:
    return SteadyStateLogicSignal(
        value=jnp.asarray(value, dtype=jnp.complex64)
    )
def logic_signal(mode=STEADY_STATE, **kwargs):
    if mode==STEADY_STATE:
        return _steady_state_logic_signal(**kwargs)
    elif mode==TEMPORAL:
        return None

def complete_steady_state_inputs(
    inputs: dict[str, SteadyStateOpticalSignal|SteadyStateElectricalSignal|SteadyStateLogicSignal]
):
    optical_wls = [jnp.array([])]
    electrical_wls = [jnp.array([])]
    num_wls_per_port = {}
    for port, signal in inputs.items():
        if isinstance(signal, SteadyStateOpticalSignal):
            optical_wls.append(signal.wl)
        elif isinstance(signal, SteadyStateElectricalSignal):
            electrical_wls.append(signal.wl)
            # num_wls_per_port[port] = signal.wl.shape[0]
    
    optical_wls = jnp.unique(jnp.concatenate(optical_wls))
    electrical_wls = jnp.unique(jnp.concatenate(electrical_wls))
    for port, signal in inputs.items():
        inputs[port] = _complete_steady_state_signal(signal, optical_wls, electrical_wls)
    pass

def _complete_steady_state_signal(
    signal: SteadyStateOpticalSignal|SteadyStateElectricalSignal|SteadyStateLogicSignal,
    optical_wls,
    electrical_wls,
) -> SteadyStateOpticalSignal|SteadyStateElectricalSignal|SteadyStateLogicSignal:
    if isinstance(signal, SteadyStateLogicSignal):
        return signal
    elif isinstance(signal, SteadyStateOpticalSignal):
        mask = ~jnp.isin(optical_wls, signal.wl)
        missing_wls = optical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        field = jnp.concatenate([signal.field, jnp.zeros_like(missing_wls)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        field = field[sort_idx]
        return SteadyStateOpticalSignal(field=field, wl=wl, polarization=signal.polarization)

    elif isinstance(signal, SteadyStateElectricalSignal):
        mask = ~jnp.isin(electrical_wls, signal.wl)
        missing_wls = electrical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        voltage = jnp.concatenate([signal.voltage, jnp.zeros_like(missing_wls)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        voltage = voltage[sort_idx]
        return SteadyStateElectricalSignal(voltage=voltage, wl=wl)

    

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