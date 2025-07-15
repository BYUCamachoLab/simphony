from flax import struct
import jax.numpy as jnp
from typing import Union


@struct.dataclass
class BlockModeOpticalSignal:
    field: jnp.ndarray    # shape:(N, T) where N is number of wavelengths and T is time steps
    wl: jnp.ndarray       # shape: (N,) or (N, T)
    polarization: jnp.ndarray  # shape: (N, 2) or (2,) if fixed polarization


@struct.dataclass
class BlockModeElectricalSignal:
    field: jnp.ndarray    # shape:(N, T) where N is number of wavelengths and T is time steps
    wl: jnp.ndarray       # shape: (N,) or (N, T)

@struct.dataclass
class BlockModeLogicSignal:
    value: jnp.ndarray


def block_mode_optical_signal(
    field: Union[float, complex, list, jnp.ndarray] = 0.0 + 0.0j,
    wl: Union[float, list, jnp.ndarray] = 1550e-9,
    polarization: Union[list, jnp.ndarray] = None
) -> BlockModeOpticalSignal:
    field = jnp.atleast_2d(jnp.asarray(field, dtype=jnp.complex64))
    wl = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))

    # Default polarization
    if polarization is None:
        polarization = jnp.tile(jnp.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64), (field.shape[0], wl.shape[0], 1))
    else:
        polarization = jnp.asarray(polarization, dtype=jnp.complex64)

    return BlockModeOpticalSignal(field=field, wl=wl, polarization=polarization)


def block_mode_electrical_signal(
    field: Union[float, complex, list, jnp.ndarray] = 0.0 + 0.0j,
    wl: Union[float, list, jnp.ndarray] = [0],
) -> BlockModeElectricalSignal:
    field = jnp.atleast_2d(jnp.asarray(field, dtype=jnp.complex64))
    wl = jnp.atleast_1d(jnp.asarray(wl, dtype=jnp.float32))
    return BlockModeElectricalSignal(field=field, wl=wl)

def block_mode_logic_signal(
    value: Union[float, complex, jnp.ndarray] = 0.0 + 0.0j
) -> BlockModeLogicSignal:
    return BlockModeLogicSignal(
        value=jnp.atleast_2d(jnp.asarray(value, dtype=jnp.complex64))
    )


def complete_block_mode_inputs(
    inputs: dict[str, BlockModeOpticalSignal|BlockModeElectricalSignal|BlockModeLogicSignal]
):
    optical_wls = [jnp.array([])]
    electrical_wls = [jnp.array([])]
    num_wls_per_port = {}
    for port, signal in inputs.items():
        if isinstance(signal, BlockModeOpticalSignal):
            optical_wls.append(signal.wl)
        elif isinstance(signal, BlockModeElectricalSignal):
            electrical_wls.append(signal.wl)
            # num_wls_per_port[port] = signal.wl.shape[0]
    
    optical_wls = jnp.unique(jnp.concatenate(optical_wls))
    electrical_wls = jnp.unique(jnp.concatenate(electrical_wls))
    for port, signal in inputs.items():
        inputs[port] = _complete_block_mode_signal(signal, optical_wls, electrical_wls)
    pass

def _complete_block_mode_signal(
    signal: BlockModeOpticalSignal|BlockModeElectricalSignal|BlockModeLogicSignal,
    optical_wls,
    electrical_wls,
) -> BlockModeOpticalSignal|BlockModeElectricalSignal|BlockModeLogicSignal:
    if isinstance(signal, BlockModeLogicSignal):
        return signal
    elif isinstance(signal, BlockModeOpticalSignal):
        mask = ~jnp.isin(optical_wls, signal.wl)
        missing_wls = optical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        field = jnp.concatenate([signal.field, jnp.zeros_like((missing_wls.shape[0],signal.field.shape[1]), dtype=signal.field.dtype)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        field = field[sort_idx]
        return BlockModeOpticalSignal(field=field, wl=wl, polarization=signal.polarization)

    elif isinstance(signal, BlockModeElectricalSignal):
        mask = ~jnp.isin(electrical_wls, signal.wl)
        missing_wls = electrical_wls[mask]
        wl = jnp.concatenate([signal.wl, missing_wls])
        voltage = jnp.concatenate([signal.voltage, jnp.zeros_like((missing_wls.shape[0],signal.voltage.shape[1]), dtype=signal.voltage.dtype)])
        sort_idx = jnp.argsort(wl)

        wl = wl[sort_idx]
        voltage = voltage[sort_idx]
        return BlockModeElectricalSignal(voltage=voltage, wl=wl)

