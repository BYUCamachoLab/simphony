"""Simphony components that interpret SAX S-parameter models.

`SParameterElement` is the single class Simphony uses to interpret a SAX
model. Raw SAX callables placed in a `Circuit` (or in a PCell's `models`) are
wrapped with `optical_s_parameter`, which returns a `SParameterElement`
subclass whose class-level `ports` are derived from the SAX model.

The same element serves every simulator:

- `SParameterSimulation` calls `s_parameters`, which returns the full SAX
  S-dict.
- `SampleModeSimulation` and `BlockModeSimulation` vector-fit the S-matrix
  into a discrete state-space model and do the port/mode indexing
  themselves.

The circuit assigns each element instance a port directionality after the
netlist is flattened (see `simphony.circuit.circuit.InstantiatedCircuit`).
Directionality only defines which way is the simulator's forward pass; the
underlying SAX model keeps all of its bidirectional information.
"""

import functools
import inspect
import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
import sax
from jax.typing import ArrayLike
from sax import DEFAULT_MODES
from scipy.constants import speed_of_light

from simphony.component.component import (
    BlockModeComponent,
    SampleModeComponent,
    SParameterComponent,
)
from simphony.component.port import Port
from simphony.signal.block_mode import BlockModeOpticalSignal
from simphony.signal.sample_mode import SampleModeOpticalSignal
from simphony.simulation.simulation import SimulationMode, SimulationParameters
from simphony.time_domain.vector_fitting.passivity import (
    enforce_passivity_discrete,
    passivity_assessment_discrete,
)
from simphony.time_domain.vector_fitting.order_estimation import estimate_order_bounds
from simphony.time_domain.vector_fitting.s_domain import vector_fitting_continuous
from simphony.time_domain.vector_fitting.z_domain import (
    PHYSICIST,
    bilinear_transform,
    optimize_order_vector_fitting_discrete,
    state_space_discrete,
    state_space_discrete_optimized_terms,
    state_space_response_discrete,
    state_space_response_discrete_optimized,
    state_space_step_discrete_optimized,
    vector_fitting_discrete,
)
from simphony.utils import dict_to_rect_matrix

_default_vector_fitting_parameters = {
    "model_order": None,
    "min_model_order": None,
    "max_model_order": None,
    "min_order_tolerance": 1e-3,
    "max_order_tolerance": 1e-6,
    "num_frequency_samples": 1000,
    "center_wavelength": 1.55e-6,
    "spectral_range": (1.5e-6, 1.6e-6),
    "fitter": "z",
    "passivity": None,
    "passivity_aware_order": False,
    "passivity_options": None,
}
"""`fitter` is "z" (z-domain vector fitting, the default) or "s_bilinear" (continuous s-domain vector
fitting at baseband, then the bilinear transform at the region's sample period; needs `model_order`).

`passivity` is None (default: no check), "check" (warn if the discrete model has sigma_max > 1 anywhere on the
unit circle, i.e. anywhere in the simulated band) or "enforce" (perturb its residues to make it passive; see
`simphony.time_domain.vector_fitting.passivity`).

`passivity_aware_order` (default False) applies when the model order is searched (`model_order=None`) and
`passivity="enforce"`: True scores each candidate order by its error after enforcement, not before, so the search
does not favour orders that fit well only until enforcement changes them. Opt-in: in a study of five SiEPIC-based
elements it changed the selected order in only 2 of 10 cases (time_domain_paper/passivity_order).

`passivity_options` is None or a dict of keyword options for `enforce_passivity_discrete` (for example
`out_of_band_weight`, `margin`, `max_iterations`), used both by the order search and by the final enforcement.

Model order: a fixed `model_order`, or a golden-section search between `min_model_order` and `max_model_order`. A
bound left as None (the default) is estimated a priori from the data (`order_estimation.estimate_order_bounds`): the
count of normalised Loewner singular values above `min_order_tolerance` (default 1e-3, about a -60 dB maximum fit
error) for the minimum, and above `max_order_tolerance` (default 1e-6, about the data's accuracy floor) for the
maximum. This puts the search past the error cliff of responses with many fringes or resonances, where the search is
well conditioned. An explicit `min_model_order` below the estimate is used as given, with a warning.

References: golden-section order search, Rosenthal & Vick, EMC Europe 2023,
https://doi.org/10.1109/EMCEurope57790.2023.10274284; Loewner order estimation, Mayo & Antoulas 2007,
https://doi.org/10.1016/j.laa.2007.03.008, and Lefteriu & Antoulas 2010, https://doi.org/10.1109/TCAD.2009.2034500
(full list in `simphony.time_domain.vector_fitting.order_estimation`)."""

S_PARAMETER_ELEMENT_SETTING_KEYS = {
    "sax_settings",
    "vector_fitting_parameters",
    "delay_compensation",
    "apply_phase_correction",
    "port_directionality",
    "s_parameter_group",
}
"""Settings understood by `SParameterElement`. When an instance's settings
contain none of these keys, they are interpreted as SAX settings."""

NO_S_PARAMETER_GROUP = -1
"""`s_parameter_group` value that prevents an element from being fused."""

MIN_FIT_OVERSAMPLING = 1.5
"""A vector-fitted element warns when its region's sample rate is less than
this many times the fitted spectral range."""


def normalize_vector_fitting_parameters(vector_fitting_parameters=None) -> dict:
    """Return a complete copy of `vector_fitting_parameters`.

    Missing keys are filled from the defaults. A fixed `model_order`
    disables the model-order search (`min_model_order` and
    `max_model_order` become `None`).
    """
    parameters = deepcopy(_default_vector_fitting_parameters)
    parameters.update(deepcopy(vector_fitting_parameters or {}))
    if parameters["model_order"] is not None:
        parameters["min_model_order"] = None
        parameters["max_model_order"] = None
    return parameters


class SParameterElement(SParameterComponent, SampleModeComponent, BlockModeComponent):
    """A Simphony component that interprets a SAX S-parameter model.

    Do not subclass this directly; use `optical_s_parameter` to create a
    subclass for a specific SAX model.

    Instance settings
    -----------------
    sax_settings:
        Keyword arguments passed to the SAX model.
    vector_fitting_parameters:
        Partial or complete vector-fitting parameters, merged over the
        defaults (`model_order`, `min_model_order`, `max_model_order`,
        `num_frequency_samples`, `center_wavelength`, `spectral_range`,
        `fitter`, `passivity`, `passivity_aware_order`, `passivity_options`,
        `min_order_tolerance`, `max_order_tolerance`). Unset order bounds are
        estimated a priori from the data.
    delay_compensation:
        Sample mode only. Number of samples of artificial delay to remove from
        the fitted model. Ignored (with a warning) in block mode.
    apply_phase_correction:
        Sample mode only. Whether the delay compensation phase is applied to
        the outputs.
    port_directionality:
        Mapping from port name to `"input"`, `"output"`, or
        `"bidirectional"`. Overrides the factory default and any direction
        inferred from the netlist.
    s_parameter_group:
        Integer used when fusing adjacent elements in block mode. Only
        adjacent elements with the same group fuse; `-1` never fuses.
    """

    _sax_model = None
    _default_modes = DEFAULT_MODES
    ports = []

    def __repr__(self):
        return f"<{type(self).__name__} (SParameterElement obj)>"

    def __init__(self, simulation_parameters: SimulationParameters, **settings):
        unknown = set(settings) - S_PARAMETER_ELEMENT_SETTING_KEYS
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unknown settings {sorted(unknown)}. "
                "SAX model keyword arguments must go under 'sax_settings' when "
                f"any of {sorted(S_PARAMETER_ELEMENT_SETTING_KEYS)} is given."
            )
        self.sax_model = type(self)._sax_model
        self.settings = deepcopy(settings)
        self.settings.setdefault("sax_settings", {})
        self.settings["vector_fitting_parameters"] = (
            normalize_vector_fitting_parameters(
                self.settings.get("vector_fitting_parameters")
            )
        )
        self.settings.setdefault("delay_compensation", 0)
        self.settings.setdefault("apply_phase_correction", True)
        self.settings.setdefault("port_directionality", {})
        self.settings.setdefault("s_parameter_group", 0)

        if self.settings["apply_phase_correction"]:
            self._k = self.settings["delay_compensation"]
        else:
            self._k = 0

        self._state_space_cache = {}
        self.set_port_directionality(self.settings["port_directionality"])

    # ------------------------------------------------------------------ ports
    @property
    def port_directionality(self) -> dict:
        return {p.name: p.directionality for p in self.ports}

    def set_port_directionality(self, port_directionality: dict):
        """Set the directionality of this instance's ports.

        Ports not named in `port_directionality` keep their current
        directionality. This only affects this instance, not the class.
        """
        unknown = set(port_directionality) - {p.name for p in self.ports}
        if unknown:
            raise ValueError(
                f"{type(self).__name__} has no ports named {sorted(unknown)}"
            )
        self.ports = [
            Port(
                name=p.name,
                type=p.type,
                directionality=port_directionality.get(p.name, p.directionality),
            )
            for p in self.ports
        ]
        self._create_port_lookup_table()
        self._state_space_cache = {}

    # ---------------------------------------------------------- S-parameters
    def s_parameters(
        self,
        inputs: dict,
        wl: ArrayLike = 1.55e-6,
    ):
        if _has_wl_kwarg(self.sax_model):
            return self.sax_model(wl=wl, **self.settings["sax_settings"])
        return self.sax_model(**self.settings["sax_settings"])

    # ------------------------------------------------------------ state space
    def _state_space(self, simulation_parameters):
        """Vector-fit the SAX model (memoized) for the active simulator.

        In block mode without the backward pass, only the forward
        `S[output <- input]` relationships are fitted. Otherwise, the
        relationships allowed by the port directionality are fitted; for
        block mode with the backward pass this is the full S-matrix.

        Returns a dict with the state-space matrices `A, B, C, D`, and
        `inputs`/`outputs`: lists of `(port_name, mode_index)` for each
        state-space column/row.
        """
        block_mode = simulation_parameters.simulation_mode == SimulationMode.BLOCK_MODE
        if block_mode and getattr(simulation_parameters, "backward_pass", False):
            directionality = {p.name: "bidirectional" for p in self.ports}
        else:
            directionality = self.port_directionality

        if block_mode:
            if self.settings["delay_compensation"] != 0:
                warnings.warn(
                    f"{type(self).__name__}: delay_compensation is ignored in "
                    "block mode simulations"
                )
            delay_compensation = 0
        else:
            delay_compensation = self.settings["delay_compensation"]

        mode_identifiers = tuple(simulation_parameters.mode_identifiers)
        key = (
            tuple(sorted(directionality.items())),
            delay_compensation,
            float(simulation_parameters.dt),
            mode_identifiers,
        )
        if key in self._state_space_cache:
            return self._state_space_cache[key]

        # Only wavelength-dependent models are vector-fitted; a model without
        # a `wl` argument becomes a constant feedthrough matrix directly.
        if _has_wl_kwarg(self.sax_model):
            span = speed_of_light / min(
                self.settings["vector_fitting_parameters"]["spectral_range"]
            ) - speed_of_light / max(
                self.settings["vector_fitting_parameters"]["spectral_range"]
            )
            oversampling = 1 / (span * simulation_parameters.dt)
            if oversampling < MIN_FIT_OVERSAMPLING:
                warnings.warn(
                    f"{type(self).__name__}: the sample rate of this element's region "
                    f"({1 / simulation_parameters.dt / 1e12:.3g} THz) is only "
                    f"{oversampling:.2f}x the fitted spectral range "
                    f"({span / 1e12:.3g} THz); the z-domain vector fit is unreliable "
                    f"below {MIN_FIT_OVERSAMPLING}x"
                    + (", and below 1x the z-domain model will alias" if oversampling < 1 else "")
                    + ". Narrow `spectral_range` or run the element at a higher "
                    "sample rate (e.g. move it to a faster region)."
                )
        filtered_model = _get_filtered_sax_model(
            self.sax_model, directionality, mode_identifiers
        )
        (
            (A, B, C, D),
            input_ports,
            output_ports,
        ) = _calculate_state_space_coefficients_from_sax_model(
            filtered_model,
            self.settings["sax_settings"],
            self.settings["vector_fitting_parameters"],
            simulation_parameters,
            delay_compensation=delay_compensation,
        )

        def port_mode_indices(port_modes):
            indices = []
            for port_mode in port_modes:
                port, mode = port_mode.split("@")
                indices.append((port, _mode_index(mode, mode_identifiers)))
            return indices

        state_space = {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "inputs": port_mode_indices(input_ports),
            "outputs": port_mode_indices(output_ports),
            "input_ports": input_ports,
            "output_ports": output_ports,
        }
        self._state_space_cache[key] = state_space
        return state_space

    def _baseband_delta_omega(self, simulation_parameters):
        """Per-wavelength rotation (rad/sample) from the fit center."""
        wl_center = self.settings["vector_fitting_parameters"]["center_wavelength"]
        wls = simulation_parameters.optical_baseband_wavelengths
        return (
            2
            * jnp.pi
            * speed_of_light
            * (1.0 / wls - 1.0 / wl_center)
            * simulation_parameters.dt
        )

    # ------------------------------------------------------------ block mode
    def block_mode_response(self, input_signals: dict, simulation_parameters):
        """Evaluate the fitted state-space model over a full time block.

        Every `(port, mode)` pair of the fitted model is a separate
        state-space input/output. Missing input signals are treated as zero.
        Outputs are returned for every port the fitted model can emit on;
        entries for input ports are backward-travelling waves (reflections).
        """
        state_space = self._state_space(simulation_parameters)
        A, B, C, D = (state_space[k] for k in "ABCD")

        wls = simulation_parameters.optical_baseband_wavelengths
        T = simulation_parameters.num_time_steps
        L = wls.shape[0]
        M = len(simulation_parameters.mode_identifiers)

        u = jnp.zeros((T, L, len(state_space["inputs"])), dtype=complex)
        for column, (port, mode_idx) in enumerate(state_space["inputs"]):
            signal = input_signals.get(port)
            if signal is not None:
                u = u.at[:, :, column].set(signal.amplitude[:, :, mode_idx])

        update_constant = jnp.exp(
            1j * self._baseband_delta_omega(simulation_parameters)
        )
        if simulation_parameters.use_state_space_optimization:
            y, _ = state_space_response_discrete_optimized(
                A, B, C, D, update_constant, u
            )
        else:

            def single_wavelength(phase, u_l):
                y_l, _ = state_space_response_discrete(phase * A, phase * B, C, D, u_l)
                return y_l

            y = jax.vmap(single_wavelength, in_axes=(0, 1), out_axes=1)(
                update_constant, u
            )

        amplitudes = {}
        for row, (port, mode_idx) in enumerate(state_space["outputs"]):
            amplitude = amplitudes.get(port, jnp.zeros((T, L, M), dtype=complex))
            amplitudes[port] = amplitude.at[:, :, mode_idx].set(y[:, :, row])

        return {
            port: BlockModeOpticalSignal(amplitude=amplitude, wavelength=wls)
            for port, amplitude in amplitudes.items()
        }

    # ----------------------------------------------------------- sample mode
    def sample_mode_initial_state(self, simulation_parameters):
        state_space = self._state_space(simulation_parameters)
        A, B, C, D = (state_space[k] for k in "ABCD")
        self.state_space_matrices = (A, B, C, D)
        self.state_space_input_indices = {
            tuple(p.split("@")): i for i, p in enumerate(state_space["input_ports"])
        }
        self.state_space_output_indices = {
            tuple(p.split("@")): i for i, p in enumerate(state_space["output_ports"])
        }
        self._sample_mode_inputs = state_space["inputs"]
        self._sample_mode_outputs = state_space["outputs"]

        self._optimized_state_space_terms = None
        if simulation_parameters.use_state_space_optimization:
            try:
                self._optimized_state_space_terms = (
                    state_space_discrete_optimized_terms(A, B, C)
                )
            except ValueError:
                self._optimized_state_space_terms = None

        L = len(simulation_parameters.optical_baseband_wavelengths)
        return jnp.zeros((L, A.shape[1]), dtype=complex)

    def sample_mode_step(
        self,
        input_signals: dict,
        state: jax.Array,
        simulation_state,
        simulation_parameters,
    ):
        """Compute the next state of the system.

        For each wavelength `l` (vectorized over wavelengths):
            new_x[l] = phase_AB[l] * (A @ x[l] + B @ u[l])
            y[l]     = phase_CD[l] * (C @ x[l] + D @ u[l])
        """
        x = state
        A, B, C, D = self.state_space_matrices

        L = len(simulation_parameters.optical_baseband_wavelengths)
        M = len(simulation_parameters.mode_identifiers)
        u = jnp.zeros((L, len(self._sample_mode_inputs)), dtype=complex)
        for column, (port_name, mode_idx) in enumerate(self._sample_mode_inputs):
            u = u.at[:, column].set(input_signals[port_name].amplitude[:, mode_idx])

        delta_omega = self._baseband_delta_omega(simulation_parameters)
        phase_AB = jnp.exp(1j * delta_omega)
        phase_CD = jnp.exp(1j * self._k * delta_omega)

        if (
            simulation_parameters.use_state_space_optimization
            and self._optimized_state_space_terms is not None
        ):
            A_diag, C_opt = self._optimized_state_space_terms
            y, new_x = state_space_step_discrete_optimized(
                A_diag, C_opt, D, phase_AB, u, x
            )
            y = phase_CD[:, None] * y
        else:
            new_x = phase_AB[:, None] * (x @ A.T + u @ B.T)
            y = phase_CD[:, None] * (x @ C.T + u @ D.T)

        output_signals = {
            port_name: SampleModeOpticalSignal(
                amplitude=jnp.zeros((L, M), dtype=complex),
                wavelength=simulation_parameters.optical_baseband_wavelengths,
            )
            for port_name in self._output_optical_port_names
        }
        for row, (port_name, mode_idx) in enumerate(self._sample_mode_outputs):
            if port_name not in output_signals:
                continue
            signal = output_signals[port_name]
            output_signals[port_name] = signal.replace(
                amplitude=signal.amplitude.at[:, mode_idx].set(y[:, row])
            )

        return output_signals, new_x


def optical_s_parameter(
    sax_model: sax.Model,
    port_directionality: dict = None,
    default_modes: list | tuple | str = DEFAULT_MODES,
) -> type[SParameterElement]:
    """Wrap a SAX optical model as a Simphony `SParameterElement` class.

    Parameters
    ----------
    sax_model:
        Callable SAX model returning an `SDict`.
    port_directionality:
        Optional default mapping from port name to `"input"`, `"output"`, or
        `"bidirectional"`. Unspecified ports default to `"bidirectional"`, in
        which case directed simulators infer the direction from the netlist.
        Instance settings may override these defaults.
    default_modes:
        Mode label or labels used when the SAX model does not encode
        explicit multimode port names.

    Returns
    -------
    type[SParameterElement]
        A component class; see `SParameterElement` for its settings.
    """
    port_names = sorted(_get_port_names_without_mode(sax_model))
    port_directionality = dict(port_directionality or {})
    unknown = set(port_directionality) - set(port_names)
    if unknown:
        raise ValueError(f"SAX model has no ports named {sorted(unknown)}")

    if isinstance(default_modes, str):
        default_modes = [default_modes]

    name = getattr(sax_model, "__name__", "sax_model")
    return type(
        f"SParameterElement_{name}",
        (SParameterElement,),
        {
            "_sax_model": staticmethod(sax_model),
            "_default_modes": tuple(default_modes),
            "ports": [
                Port(
                    name=port_name,
                    type="optical",
                    directionality=port_directionality.get(port_name, "bidirectional"),
                )
                for port_name in port_names
            ],
        },
    )


def wrap_sax_settings(instance_names, get_model, settings: dict) -> dict:
    """Interpret plain SAX keyword settings of `SParameterElement` instances.

    Settings of an `SParameterElement` instance that contain none of the
    `S_PARAMETER_ELEMENT_SETTING_KEYS` are treated as SAX settings and moved
    under `"sax_settings"`. Returns a new settings dict.
    """
    settings = dict(settings)
    for instance_name in instance_names:
        model = get_model(instance_name)
        if not (inspect.isclass(model) and issubclass(model, SParameterElement)):
            continue
        instance_settings = settings.get(instance_name, {})
        if not S_PARAMETER_ELEMENT_SETTING_KEYS & set(instance_settings):
            settings[instance_name] = {"sax_settings": dict(instance_settings)}
    return settings


# ------------------------------------------------------------------- fusing
def fused_sax_model(members: dict, netlist: dict, mode_identifiers):
    """Build one SAX model for a group of connected `SParameterElement`s.

    Parameters
    ----------
    members:
        Mapping from member instance name to the instantiated element.
    netlist:
        SAX netlist (`instances`, `connections`, `ports`) of the group, keyed
        by the (sanitized) member instance names.
    mode_identifiers:
        Modes the member models are expanded to.

    Each member's `sax_settings` are bound, so the returned model only takes
    `wl`. Members are used unfiltered, so reflections inside the group are
    kept.
    """
    models = {}
    for instance_name, element in members.items():
        member_model = functools.partial(
            element.sax_model, **element.settings["sax_settings"]
        )
        if not _has_wl_kwarg(element.sax_model):
            member_model = _ignore_wl(member_model)
        models[instance_name] = _multimode_model(member_model, mode_identifiers)

    netlist = {
        "instances": {k: k for k in netlist["instances"]},
        "connections": netlist["connections"],
        "ports": netlist["ports"],
    }
    circuit, _ = sax.circuit(netlist, models)

    def fused_model(wl=1.55):
        return circuit(wl=wl)

    return fused_model


def merge_vector_fitting_parameters(group_label, members: dict) -> dict:
    """Merge member vector-fitting parameters for a fused element.

    `spectral_range` is the union (covering interval) of the members'
    ranges, and the model order is searched between the smallest minimum and
    largest maximum order (a fixed `model_order=n` counts as `n..n`).
    `num_frequency_samples`, `center_wavelength`, `fitter`, `passivity`, `passivity_aware_order`,
    `passivity_options`, `min_order_tolerance` and `max_order_tolerance` must be identical, as
    must `apply_phase_correction`.
    """
    parameters = [
        (name, element.settings["vector_fitting_parameters"])
        for name, element in members.items()
    ]
    for key in ("num_frequency_samples", "center_wavelength", "fitter", "passivity", "passivity_aware_order",
                "passivity_options", "min_order_tolerance", "max_order_tolerance"):
        values = {name: repr(p[key]) for name, p in parameters}
        if len(set(values.values())) > 1:
            raise ValueError(
                f"Cannot fuse S-parameter group {group_label}: members disagree on "
                f"vector_fitting_parameters['{key}']: {values}"
            )
    apply_phase_correction = {
        name: element.settings["apply_phase_correction"]
        for name, element in members.items()
    }
    if len(set(apply_phase_correction.values())) > 1:
        raise ValueError(
            f"Cannot fuse S-parameter group {group_label}: members disagree on "
            f"apply_phase_correction: {apply_phase_correction}"
        )

    def order_bounds(p):
        if p["model_order"] is not None:
            return p["model_order"], p["model_order"]
        return p["min_model_order"], p["max_model_order"]

    bounds = [order_bounds(p) for _, p in parameters]
    # a bound left as None (estimate a priori) by any member is estimated for the fused data
    lows, highs = [b[0] for b in bounds], [b[1] for b in bounds]
    min_order = None if None in lows else min(lows)
    max_order = None if None in highs else max(highs)
    ranges = [p["spectral_range"] for _, p in parameters]

    merged = deepcopy(parameters[0][1])
    merged["spectral_range"] = (
        min(min(r) for r in ranges),
        max(max(r) for r in ranges),
    )
    if min_order is not None and min_order == max_order:
        merged.update(model_order=min_order, min_model_order=None, max_model_order=None)
    else:
        merged.update(
            model_order=None, min_model_order=min_order, max_model_order=max_order
        )
    return merged


# ------------------------------------------------------------------ helpers
def _ignore_wl(model):
    @functools.wraps(model)
    def wrapped(wl=1.55, **kwargs):
        return model(**kwargs)

    return wrapped


def _multimode_model(model, modes):
    def wrapped(wl=1.55):
        return sax.multimode(model(wl=wl), modes=tuple(modes))

    return wrapped


def _mode_index(mode, mode_identifiers) -> int:
    if mode in mode_identifiers:
        return mode_identifiers.index(mode)
    normalized = [_normalize_mode_name(m) for m in mode_identifiers]
    return normalized.index(_normalize_mode_name(mode))


def _has_wl_kwarg(model) -> bool:
    try:
        if "wl" in inspect.signature(model).parameters:
            return True
    except (TypeError, ValueError):
        pass
    try:
        model(wl=0.0)
        return True
    except TypeError:
        return False


def _get_port_names_without_mode(sax_model):
    sdict = sax.multimode(sax_model())
    port_names = set()
    for in_portmode, out_portmode in sdict.keys():
        port_names.add(in_portmode.split("@")[0])
        port_names.add(out_portmode.split("@")[0])
    return port_names


def _get_filtered_sax_model(
    sax_model: sax.Model,
    port_directionality,
    default_modes,
):
    """Return a multimode SAX model keeping only relationships allowed by
    `port_directionality` (no waves into output ports, none out of input
    ports)."""
    input_ports_to_remove = {
        port_name
        for port_name, direction in port_directionality.items()
        if direction == "output"
    }
    output_ports_to_remove = {
        port_name
        for port_name, direction in port_directionality.items()
        if direction == "input"
    }

    @functools.wraps(sax_model)
    def filtered_sax_model(*args, **kwargs):
        sdict = sax.multimode(sax_model(*args, **kwargs), modes=default_modes)

        def is_allowed(key):
            dst, src = key
            return (
                src.split("@")[0] not in input_ports_to_remove
                and dst.split("@")[0] not in output_ports_to_remove
            )

        return {k: v for k, v in sdict.items() if is_allowed(k)}

    filtered_sax_model.__signature__ = inspect.signature(sax_model)
    return filtered_sax_model


def _normalize_mode_name(mode) -> str:
    return str(mode).lower()


def _ordered_from_set(mode_set, mode_identifiers=None):
    if mode_identifiers is None:
        return tuple(sorted(mode_set, key=_normalize_mode_name))

    wanted = [_normalize_mode_name(m) for m in mode_identifiers]
    present = {_normalize_mode_name(m): m for m in mode_set}

    ordered = [present[m] for m in wanted if m in present]
    extras = [m for m in mode_set if _normalize_mode_name(m) not in set(wanted)]
    ordered.extend(sorted(extras, key=_normalize_mode_name))
    return tuple(ordered)


def _get_port_mode_luts(
    sax_model: sax.Model,
    mode_identifiers=None,
):
    input_port_modes = {}
    output_port_modes = {}

    for o, i in sax_model().keys():
        in_port, in_mode = i.split("@")
        out_port, out_mode = o.split("@")
        input_port_modes.setdefault(in_port, set()).add(in_mode)
        output_port_modes.setdefault(out_port, set()).add(out_mode)

    input_port_modes = {
        port: _ordered_from_set(modes, mode_identifiers)
        for port, modes in input_port_modes.items()
    }
    output_port_modes = {
        port: _ordered_from_set(modes, mode_identifiers)
        for port, modes in output_port_modes.items()
    }

    return input_port_modes, output_port_modes


def _calculate_state_space_coefficients_from_sax_model(
    sax_model,
    sax_settings,
    vector_fitting_parameters,
    simulation_parameters,
    delay_compensation=0,
):
    """Returns (A, B, C, D), input_ports, output_ports.

    For D[i, j], output_ports[i] <- input_ports[j]. Port names have the
    form "port@mode".
    """
    vector_fitting_parameters = normalize_vector_fitting_parameters(
        vector_fitting_parameters
    )
    input_port_modes, output_port_modes = _get_port_mode_luts(
        sax_model, simulation_parameters.mode_identifiers
    )
    input_ports = [
        f"{port}@{mode}" for port, modes in input_port_modes.items() for mode in modes
    ]
    output_ports = [
        f"{port}@{mode}" for port, modes in output_port_modes.items() for mode in modes
    ]

    if not _has_wl_kwarg(sax_model):
        if not delay_compensation == 0:
            raise ValueError(
                "delay compensation cannot be applied to a 0 delay element"
            )

        sdict = sax_model(**sax_settings)
        S = dict_to_rect_matrix(
            sdict, input_ports=input_ports, output_ports=output_ports
        )
        m = len(input_ports)
        q = len(output_ports)
        A = jnp.zeros((m, m), dtype=complex)
        B = jnp.zeros((m, m), dtype=complex)
        C = jnp.zeros((q, m), dtype=complex)
        D = S[0, :, :]

        return (A, B, C, D), input_ports, output_ports

    f_min = speed_of_light / max(vector_fitting_parameters["spectral_range"])
    f_max = speed_of_light / min(vector_fitting_parameters["spectral_range"])
    f_center = speed_of_light / vector_fitting_parameters["center_wavelength"]
    frequency = jnp.linspace(
        f_min, f_max, vector_fitting_parameters["num_frequency_samples"]
    )
    sdict = sax_model(wl=1e6 * speed_of_light / frequency, **sax_settings)

    s_params = dict_to_rect_matrix(
        sdict, input_ports=input_ports, output_ports=output_ports
    )
    sampling_frequency = 1 / simulation_parameters.dt

    Omega = 2 * jnp.pi * (frequency - f_center) / sampling_frequency
    s_params = jnp.exp(-1j * delay_compensation * Omega)[:, None, None] * s_params
    fitter = vector_fitting_parameters["fitter"]
    if fitter == "s_bilinear":
        if vector_fitting_parameters["model_order"] is None:
            raise ValueError("fitter 's_bilinear' needs an explicit model_order")
        # s = -j omega (physicist) maps by the bilinear transform onto z = e^{-j Omega}, the convention used below
        s_poles, s_residues, s_feedthrough, _ = vector_fitting_continuous(
            vector_fitting_parameters["model_order"],
            s_params,
            frequency,
            f_center,
            sign_convention=PHYSICIST,
        )
        poles, residues, feedthrough = bilinear_transform(
            s_poles, s_residues, s_feedthrough, sampling_frequency
        )
    elif fitter != "z":
        raise ValueError(f"unknown fitter {fitter!r}; use 'z' or 's_bilinear'")
    elif vector_fitting_parameters["model_order"] is None:
        min_order, max_order = _order_bounds(
            vector_fitting_parameters, s_params, frequency, f_center, sampling_frequency, input_ports, output_ports
        )
        aware = (
            vector_fitting_parameters["passivity"] == "enforce"
            and vector_fitting_parameters["passivity_aware_order"]
        )
        (
            poles,
            residues,
            feedthrough,
            mean_squared_error,
        ) = optimize_order_vector_fitting_discrete(
            min_order,
            max_order,
            s_params,
            frequency,
            f_center,
            sampling_frequency,
            sign_convention=PHYSICIST,
            passivity_enforcement=dict(vector_fitting_parameters["passivity_options"] or {}) if aware else None,
        )
    else:
        poles, residues, feedthrough, mean_squared_error = vector_fitting_discrete(
            vector_fitting_parameters["model_order"],
            s_params,
            frequency,
            f_center,
            sampling_frequency,
            sign_convention=PHYSICIST,
        )

    passivity = vector_fitting_parameters["passivity"]
    if passivity is not None:
        poles, residues, feedthrough = _apply_passivity(
            passivity, poles, residues, feedthrough, Omega, input_ports, output_ports,
            vector_fitting_parameters["passivity_options"],
        )

    A, B, C, D = state_space_discrete(poles, residues, feedthrough)

    return (A, B, C, D), input_ports, output_ports


def _order_bounds(vector_fitting_parameters, s_params, frequency, f_center, sampling_frequency, input_ports,
                  output_ports):
    """(min, max) model order for the search: explicit bounds as given, missing ones estimated a priori from the data
    (Loewner singular values; see `order_estimation`). Warns when an explicit minimum is below the estimate."""
    lo, hi = vector_fitting_parameters["min_model_order"], vector_fitting_parameters["max_model_order"]
    estimate = estimate_order_bounds(
        s_params,
        frequency,
        f_center,
        sampling_frequency,
        sign_convention=PHYSICIST,
        min_tolerance=vector_fitting_parameters["min_order_tolerance"],
        max_tolerance=vector_fitting_parameters["max_order_tolerance"],
    )
    if lo is not None and lo < estimate.min_order:
        warnings.warn(
            f"{len(output_ports)}x{len(input_ports)} model: min_model_order={lo} is below the a-priori estimate "
            f"{estimate.min_order} (Loewner singular values above {vector_fitting_parameters['min_order_tolerance']:g}; "
            f"{estimate.phase_cycles:.0f} phase cycles across the band). The order search can stop in a false valley "
            "below the order the data needs; consider min_model_order=None to use the estimate."
        )
    lo = estimate.min_order if lo is None else lo
    hi = max(estimate.max_order, lo + 2) if hi is None else hi
    if hi < lo + 2:
        raise ValueError(f"max_model_order ({hi}) must be at least min_model_order + 2 ({lo + 2})")
    return int(lo), int(hi)


def _apply_passivity(passivity, poles, residues, feedthrough, fit_omega, input_ports, output_ports, options=None):
    """`vector_fitting_parameters["passivity"]`: "check" warns, "enforce" corrects (poles kept)."""
    if passivity not in ("check", "enforce"):
        raise ValueError(f"unknown passivity option {passivity!r}; use None, 'check' or 'enforce'")
    ports = f"{len(output_ports)}x{len(input_ports)} model ({', '.join(input_ports)} -> {', '.join(output_ports)})"
    if passivity == "check":
        report = passivity_assessment_discrete(poles, residues, feedthrough, sign_convention=PHYSICIST)
        if not report.is_passive:
            warnings.warn(
                f"{ports} is not passive: sigma_max = {report.max_singular_value:.6g} at "
                f"Omega = {report.max_omega:+.4f} rad/sample ({len(report.violation_omegas)} violation peaks). "
                "Use vector_fitting_parameters['passivity'] = 'enforce' to correct it."
            )
        return poles, residues, feedthrough
    new_residues, new_feedthrough, info = enforce_passivity_discrete(
        poles, residues, feedthrough, fit_omega, sign_convention=PHYSICIST, **(options or {})
    )
    if not info["passive"]:
        warnings.warn(
            f"{ports}: passivity enforcement stopped at sigma_max = {info['max_singular_value']:.6g} "
            f"(was {info['max_singular_value_before']:.6g}) after {info['iterations']} iterations."
        )
    return poles, jnp.asarray(new_residues), jnp.asarray(new_feedthrough)
