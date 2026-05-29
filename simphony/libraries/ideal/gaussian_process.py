"""Gaussian process components for GaussianProcessSimulation.

Provides:
  - GaussianProcessCWSource      — constant-wave optical source with amplitude noise
  - LTIGaussianProcessSystem     — flat component driven by a precomputed state-space
  - gaussian_process_s_parameter — factory: SAX model → GP-compatible component class

The factory mirrors `optical_s_parameter` from s_parameters.py but skips the
mode-converter / demultiplexer sub-netlist entirely.  Port-to-state-space-column
mapping is handled inline using the "@mode" suffix conventions that
`_calculate_state_space_coefficients_from_sax_model` already produces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from sax import DEFAULT_MODES
from scipy.constants import speed_of_light

from simphony.component.component import GaussianProcessComponent
from simphony.component.port import Port

# Private helpers reused from s_parameters.py — no re-implementation needed.
from simphony.libraries.ideal.s_parameters import (
    _calculate_state_space_coefficients_from_sax_model,
    _default_vector_fitting_parameters,
    _get_filtered_sax_model,
    _get_port_names_without_mode,
)
from simphony.signal.gaussian_process import GaussianProcessOpticalSignal
from simphony.time_domain.stochastic.gaussian_process import (
    gaussian_process_response,
    white_noise_covariance,
)
from simphony.time_domain.vector_fitting.z_domain import state_space_response_discrete

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class GaussianProcessCWSource(GaussianProcessComponent):
    """Constant-wave optical source for Gaussian process simulations.

    Generates a GaussianProcessOpticalSignal with a flat (constant) mean field
    and spectrally white amplitude noise described by `noise_power`.

    Parameters (passed via settings dict)
    --------------------------------------
    amplitude : complex
        Complex field amplitude of the CW signal (default 1.0 + 0j).
    noise_power : float
        Variance of the complex amplitude noise per polarisation mode
        (default 0.0 — noiseless coherent source).
    """

    ports = [Port(name="o0", type="optical", directionality="output")]

    def __init__(
        self,
        simulation_parameters,
        *,
        amplitude: complex = 1.0 + 0j,
        noise_power: float = 0.0,
    ):
        self.amplitude = amplitude
        self.noise_power = noise_power

    def gaussian_process_mode_response(
        self, inputs: dict, simulation_parameters
    ) -> dict:
        wl = simulation_parameters.optical_baseband_wavelengths
        T = simulation_parameters.num_time_steps
        L = wl.shape[0]
        M = len(simulation_parameters.mode_identifiers)

        mean = self.amplitude * jnp.ones((T, L, M), dtype=complex)

        # white_noise_covariance returns (T, T, M, M); broadcast over wavelengths.
        cov_single = white_noise_covariance(T, M, sigma_sq=self.noise_power)
        cov = jnp.stack([cov_single] * L, axis=0)  # (L, T, T, M, M)

        return {
            "o0": GaussianProcessOpticalSignal(
                mean_amplitude=mean,
                covariance=cov,
                wavelength=wl,
            )
        }


# ---------------------------------------------------------------------------
# LTI element
# ---------------------------------------------------------------------------


class LTIGaussianProcessSystem(GaussianProcessComponent):
    """Flat GP component backed by a discrete-time state-space model.

    Initialized with the ABCD matrices from vector fitting and the
    port-mode metadata produced by
    `_calculate_state_space_coefficients_from_sax_model`.

    For each wavelength in `simulation_parameters.optical_baseband_wavelengths`
    the component:
    1. Frequency-shifts A and B by `exp(j·delta_omega)`.
    2. Computes the K-tap impulse response h_l by sending unit impulses through
       the shifted state-space (one per input channel).
    3. Stacks input means and block-diagonal covariances from all input ports.
    4. Calls `gaussian_process_response(h_l, mu_x, Cx)`.
    5. Splits the output mean / covariance back into per-port GP signals.

    Notes
    -----
    Cross-port input covariances are assumed to be zero (independent inputs).
    Within-port mode covariances are fully preserved.

    Parameters (constructor keyword args)
    --------------------------------------
    state_space_matrices : tuple (A, B, C, D)
    input_ports  : list[str]  e.g. ["o0@TE", "o0@TM"]
    output_ports : list[str]  e.g. ["o1@TE", "o1@TM"]
    center_frequency  : float  Hz
    sampling_frequency: float  Hz
    """

    # Subclasses / factory must set `ports` as a class attribute.
    ports: list = []

    def __init__(
        self,
        simulation_parameters,
        *,
        state_space_matrices,
        input_ports: list,
        output_ports: list,
        center_frequency: float,
        sampling_frequency: float,
    ):
        self.A, self.B, self.C, self.D = state_space_matrices
        self.input_ports = input_ports  # ["o0@TE", ...]
        self.output_ports = output_ports  # ["o1@TE", ...]
        self.center_frequency = center_frequency
        self.sampling_frequency = sampling_frequency

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_port_mode(port_mode_str: str):
        """Split "port@mode" into ("port", "mode")."""
        port, mode = port_mode_str.rsplit("@", 1)
        return port, mode

    def _mode_index(self, mode: str, simulation_parameters) -> int:
        modes = list(simulation_parameters.mode_identifiers)
        # case-insensitive match
        mode_lower = mode.lower()
        for idx, m in enumerate(modes):
            if str(m).lower() == mode_lower:
                return idx
        return 0  # fallback

    def _compute_impulse_response(
        self,
        A_shifted: jax.Array,
        B_shifted: jax.Array,
        K: int,
    ) -> jax.Array:
        """H[k, n_out, n_in] via unit-impulse inputs through the state-
        space."""
        n_in = B_shifted.shape[1]
        cols = []
        for i in range(n_in):
            u = jnp.zeros((K, n_in), dtype=complex).at[0, i].set(1.0)
            y, _ = state_space_response_discrete(
                A_shifted, B_shifted, self.C, self.D, u
            )
            cols.append(y)
        return jnp.stack(cols, axis=2)  # (K, n_out, n_in)

    # ------------------------------------------------------------------
    # main response
    # ------------------------------------------------------------------

    def gaussian_process_mode_response(
        self, inputs: dict, simulation_parameters
    ) -> dict:
        wls = simulation_parameters.optical_baseband_wavelengths
        T = simulation_parameters.num_time_steps
        M = len(simulation_parameters.mode_identifiers)
        L = wls.shape[0]
        K = simulation_parameters.num_ir_taps
        fs = self.sampling_frequency
        fc = self.center_frequency

        n_in = len(self.input_ports)
        n_out = len(self.output_ports)

        # Parse port-mode lists once.
        in_pm = [self._parse_port_mode(s) for s in self.input_ports]
        out_pm = [self._parse_port_mode(s) for s in self.output_ports]

        # Collect unique output port names for accumulator initialisation.
        out_port_names = list(dict.fromkeys(pm[0] for pm in out_pm))

        # Per-output-port accumulators: (L, T, M) for mean, (L, T, T, M, M) for cov.
        out_mean_acc = {p: jnp.zeros((L, T, M), dtype=complex) for p in out_port_names}
        out_cov_acc = {
            p: jnp.zeros((L, T, T, M, M), dtype=complex) for p in out_port_names
        }

        for wl_idx, wl in enumerate(wls):
            # ---- 1. Frequency-shift state-space ----
            # wl is in metres; fc is center frequency in Hz.  Convert wl to Hz first.
            delta_omega = 2.0 * jnp.pi * (speed_of_light / wl - fc) / fs
            A_s = jnp.exp(1j * delta_omega) * self.A
            B_s = jnp.exp(1j * delta_omega) * self.B

            # ---- 2. Impulse response ----
            h_l = self._compute_impulse_response(A_s, B_s, K)  # (K, n_out, n_in)

            # ---- 3. Stack inputs → (T, n_in) mean and (T, T, n_in, n_in) cov ----
            mu_x = jnp.zeros((T, n_in), dtype=complex)
            Cx = jnp.zeros((T, T, n_in, n_in), dtype=complex)

            for col_idx, (port_name, mode) in enumerate(in_pm):
                m_idx = self._mode_index(mode, simulation_parameters)
                sig = inputs[port_name]  # GaussianProcessOpticalSignal
                mu_x = mu_x.at[:, col_idx].set(sig.mean_amplitude[:, wl_idx, m_idx])
                # Diagonal covariance block for this channel.
                # sig.covariance shape: (L, T, T, M, M)
                Cx = Cx.at[:, :, col_idx, col_idx].set(
                    sig.covariance[wl_idx, :, :, m_idx, m_idx]
                )
                # Off-diagonal mode-mode cross-covariance within same port.
                for col_idx2, (port_name2, mode2) in enumerate(in_pm):
                    if col_idx2 == col_idx or port_name2 != port_name:
                        continue
                    m_idx2 = self._mode_index(mode2, simulation_parameters)
                    Cx = Cx.at[:, :, col_idx, col_idx2].set(
                        sig.covariance[wl_idx, :, :, m_idx, m_idx2]
                    )

            # ---- 4. Papoulis propagation ----
            mu_y, Cy = gaussian_process_response(h_l, mu_x, Cx)
            # mu_y: (T, n_out),  Cy: (T, T, n_out, n_out)

            # ---- 5. Split outputs back to per-port accumulators ----
            for row_idx, (port_name, mode) in enumerate(out_pm):
                m_idx = self._mode_index(mode, simulation_parameters)
                out_mean_acc[port_name] = (
                    out_mean_acc[port_name].at[wl_idx, :, m_idx].set(mu_y[:, row_idx])
                )

                for row_idx2, (port_name2, mode2) in enumerate(out_pm):
                    if port_name2 != port_name:
                        continue
                    m_idx2 = self._mode_index(mode2, simulation_parameters)
                    out_cov_acc[port_name] = (
                        out_cov_acc[port_name]
                        .at[wl_idx, :, :, m_idx, m_idx2]
                        .set(Cy[:, :, row_idx, row_idx2])
                    )

        # Build output dict.
        outputs = {}
        for port_name in out_port_names:
            # out_mean_acc[port_name]: (L, T, M) → transpose to (T, L, M)
            mean = out_mean_acc[port_name].transpose(1, 0, 2)
            cov = out_cov_acc[port_name]  # already (L, T, T, M, M)
            outputs[port_name] = GaussianProcessOpticalSignal(
                mean_amplitude=mean,
                covariance=cov,
                wavelength=wls,
            )
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def gaussian_process_s_parameter(
    sax_model,
    port_directionality: dict = None,
    default_modes=DEFAULT_MODES,
):
    """Create a GP-compatible component class from a SAX model.

    This is the GP-mode analog of ``optical_s_parameter`` from s_parameters.py.
    Returns a *class* (not an instance) that:

    - Has ports derived from the SAX model signature.
    - On instantiation, performs vector fitting to obtain A, B, C, D matrices
      and stores them in an ``LTIGaussianProcessSystem``.
    - Implements ``gaussian_process_mode_response`` via the Papoulis equations.

    There are no sub-netlists, mode converters, or demultiplexers — the port-to
    state-space-column mapping is done inline using the "@mode" suffixes that
    ``_calculate_state_space_coefficients_from_sax_model`` already produces.

    Parameters
    ----------
    sax_model : callable
        A SAX-compatible model (e.g. ``ideal.waveguide``).
    port_directionality : dict, optional
        Mapping from port name to ``"input"``, ``"output"``, or
        ``"bidirectional"`` (default ``"bidirectional"`` for all ports).
    default_modes : tuple, optional
        Polarisation modes to include (default ``sax.DEFAULT_MODES``).

    Returns
    -------
    type
        A class that is a subclass of ``LTIGaussianProcessSystem``.

    Example
    -------
    ::

        ring = gaussian_process_s_parameter(
            ring_sax_model,
            port_directionality={"o0": "input", "o1": "output"},
        )
        circuit = Circuit(netlist=..., models={"ring": ring, ...})
    """
    if port_directionality is None:
        port_directionality = {}

    if isinstance(default_modes, str):
        default_modes = [default_modes]
    default_modes = tuple(default_modes)

    pcell_port_names = _get_port_names_without_mode(sax_model)

    class GaussianProcessSParameterElement(LTIGaussianProcessSystem):
        _sax_model = staticmethod(sax_model)
        ports = [
            Port(
                name=port_name,
                type="optical",
                directionality=port_directionality.get(port_name, "bidirectional"),
            )
            for port_name in pcell_port_names
        ]

        def __init__(self, simulation_parameters, **kwargs):
            sax_settings = kwargs.get("sax_settings", {})
            vf_params = kwargs.get(
                "vector_fitting_parameters", _default_vector_fitting_parameters
            )
            delay_comp = kwargs.get("delay_compensation", 0)

            filtered_sax = _get_filtered_sax_model(
                sax_model, port_directionality, default_modes
            )
            (
                (A, B, C, D),
                in_ports,
                out_ports,
            ) = _calculate_state_space_coefficients_from_sax_model(
                filtered_sax,
                sax_settings,
                vf_params,
                simulation_parameters,
                delay_comp,
            )

            super().__init__(
                simulation_parameters,
                state_space_matrices=(A, B, C, D),
                input_ports=in_ports,
                output_ports=out_ports,
                center_frequency=speed_of_light / vf_params["center_wavelength"],
                sampling_frequency=1.0 / simulation_parameters.dt,
            )

    GaussianProcessSParameterElement.__name__ = (
        f"GP_{getattr(sax_model, '__name__', 'model')}"
    )
    GaussianProcessSParameterElement.__qualname__ = (
        GaussianProcessSParameterElement.__name__
    )
    return GaussianProcessSParameterElement
