"""Interpolate an existing S-parameter npz to a denser wavelength grid."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_sdict(npz_path: Path) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    data = np.load(npz_path)
    wavelengths = data["wavelengths"]
    sdict: Dict[Tuple[str, str], np.ndarray] = {}
    for key in data.files:
        if key == "wavelengths":
            continue
        _, dst, src = key.split("_", 2)
        sdict[(dst, src)] = np.asarray(data[key])
    return wavelengths, sdict


def interpolate_trace(old_wl: np.ndarray, values: np.ndarray, new_wl: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(old_wl), -1)
    real = np.empty((len(new_wl), flat.shape[1]), dtype=np.float64)
    imag = np.empty_like(real)
    for col in range(flat.shape[1]):
        real[:, col] = np.interp(new_wl, old_wl, flat[:, col].real)
        imag[:, col] = np.interp(new_wl, old_wl, flat[:, col].imag)
    out = real + 1j * imag
    return out.reshape((len(new_wl),) + values.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the input npz S-dict")
    parser.add_argument("--points", type=int, default=2000, help="Number of wavelength samples in the output grid")
    parser.add_argument("--output", type=Path, help="Output npz path; default appends _interp{points}")
    args = parser.parse_args()

    src = args.source
    if args.output is None:
        args.output = src.with_name(f"{src.stem}_interp{args.points}.npz")

    old_wl, sdict = load_sdict(src)
    new_wl = np.linspace(old_wl.min(), old_wl.max(), args.points)

    out = {"wavelengths": new_wl}
    for (dst, src_port), values in sdict.items():
        interp_vals = interpolate_trace(old_wl, values.squeeze(), new_wl)
        key = f"S_{dst}_{src_port}"
        out[key] = interp_vals[:, None] if values.ndim == 2 and values.shape[1] == 1 else interp_vals

    np.savez(args.output, **out)
    print(f"Wrote interpolated S-dict to {args.output}")


if __name__ == "__main__":
    main()
