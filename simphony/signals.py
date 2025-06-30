from flax import struct
import jax.numpy as jnp
from typing import Union


@struct.dataclass
class OpticalSignal:
    # Array of complex amplitudes per wavelength
    field: jnp.ndarray  # shape: (N,) or (N, T), where N is number of wavelengths, T is time samples
    wl: jnp.ndarray     # shape: (N,), corresponding wavelengths
    polarization: jnp.ndarray  # shape: (N, 2, T) or (2,T) if fixed polarization

@struct.dataclass
class ElectricalSignal:
    voltage: jnp.ndarray  # shape: (N,) or (N, T)
    wl: jnp.ndarray       # shape: (N,)

@struct.dataclass
class LogicSignal:
    value: jnp.ndarray 


def optical_signal(**kwargs):
    return _optical_signal(**kwargs)
    
def _optical_signal(
    field: Union[float, complex, list, jnp.ndarray] = 0.0 + 0.0j,
    wl: Union[float, list, jnp.ndarray] = 1550e-9,
    polarization: Union[list, jnp.ndarray] = None
) ->OpticalSignal:
    field = jnp.atleast_1d(jnp.asarray(field, dtype=jnp.complex64))
    wl = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))

    # Default polarization
    if polarization is None:
        polarization = jnp.tile(jnp.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64), (wl.shape[0], 1))
    else:
        polarization = jnp.asarray(polarization, dtype=jnp.complex64)

    return OpticalSignal(field=field, wl=wl, polarization=polarization)


def electrical_signal(**kwargs):
    return _electrical_signal(**kwargs)

def _electrical_signal(
    voltage: Union[float, complex, list, jnp.ndarray] = [0.0 + 0.0j],
    wl: Union[float, list, jnp.ndarray] = [0],
) -> ElectricalSignal:
    voltage = jnp.asarray(voltage, dtype=jnp.complex64)
    wl = jnp.asarray(wl, dtype=jnp.float32)
    return ElectricalSignal(voltage=voltage, wl=wl)


def logic_signal(**kwargs):
    return _logic_signal(**kwargs)
    
def _logic_signal(
    value: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j
) -> LogicSignal:
    return LogicSignal(
        value=jnp.asarray(value, dtype=jnp.complex64)
    )


def complete_inputs(
    inputs: dict[str, OpticalSignal|ElectricalSignal|LogicSignal]
):
    optical_wls = [jnp.array([])]
    electrical_wls = [jnp.array([])]
    num_wls_per_port = {}
    for port, signal in inputs.items():
        if isinstance(signal, OpticalSignal):
            optical_wls.append(signal.wl)
        elif isinstance(signal, ElectricalSignal):
            electrical_wls.append(signal.wl)
            # num_wls_per_port[port] = signal.wl.shape[0]
    
    optical_wls = jnp.unique(jnp.concatenate(optical_wls))
    electrical_wls = jnp.unique(jnp.concatenate(electrical_wls))
    for port, signal in inputs.items():
        inputs[port] = _complete_signal(signal, optical_wls, electrical_wls)
    pass


def _complete_signal(
    signal: OpticalSignal|ElectricalSignal|LogicSignal,
    optical_wls,
    electrical_wls,
) -> OpticalSignal|ElectricalSignal|LogicSignal:
    if isinstance(signal, LogicSignal):
        return signal
    
    elif isinstance(signal, OpticalSignal):
        mask = ~jnp.isin(optical_wls, signal.wl)
        missing_wls = optical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        field = jnp.concatenate([signal.field, jnp.zeros((missing_wls.shape[0],signal.field.shape[1]), dtype=signal.field.dtype)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        field = field[sort_idx]
        return OpticalSignal(field=field, wl=wl, polarization=signal.polarization)

    elif isinstance(signal, ElectricalSignal):
        mask = ~jnp.isin(electrical_wls, signal.wl)
        missing_wls = electrical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        voltage = jnp.concatenate([signal.voltage, jnp.zeros((missing_wls.shape[0],signal.field.shape[1]), dtype=signal.voltage.dtype)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        voltage = voltage[sort_idx]
        return ElectricalSignal(voltage=voltage, wl=wl)

    

