"""Generic, type-agnostic operations on Simphony signals.

They act on a signal's data fields (`_data_fields`); metadata fields are left
untouched. Block-mode data fields carry time on axis 0.
"""

import jax.numpy as jnp


def data_fields(signal) -> tuple:
    """Names of the per-sample data fields of `signal`."""
    try:
        return type(signal)._data_fields
    except AttributeError as e:
        raise TypeError(
            f"{type(signal).__name__} does not declare `_data_fields`; it cannot "
            "be handled by generic multirate components"
        ) from e


def map_data_fields(signal, fn):
    """Apply `fn` to every data field of `signal`."""
    return signal.replace(**{f: fn(getattr(signal, f)) for f in data_fields(signal)})


def zeros_like_signal(signal):
    """A signal of the same type and shapes whose data fields are zero."""
    return map_data_fields(signal, jnp.zeros_like)


def decimate_block(signal, factor: int, offset: int = 0):
    """Keep samples `offset, offset + factor, ...` of a block-mode signal."""
    return map_data_fields(signal, lambda x: jnp.asarray(x)[offset::factor])


def upsample_block(signal, factor: int, offset: int = 0, mode: str = "hold"):
    """Raise the rate of a block-mode signal by `factor`.

    `mode="hold"` repeats every sample `factor` times (zero-order hold, delayed
    by `offset` samples); `mode="zeros"` places sample n at output index
    `n * factor + offset` and fills the rest with zeros.
    """
    if mode not in ("hold", "zeros"):
        raise ValueError(f"unknown upsampling mode {mode!r}")

    def upsample(x):
        x = jnp.asarray(x)
        if mode == "hold":
            y = jnp.repeat(x, factor, axis=0)
            if offset:
                pad = jnp.repeat(jnp.zeros_like(x[:1]), offset, axis=0)
                y = jnp.concatenate([pad, y[:-offset]], axis=0)
            return y
        y = jnp.zeros((x.shape[0] * factor,) + x.shape[1:], dtype=x.dtype)
        return y.at[offset::factor].set(x)

    return map_data_fields(signal, upsample)
