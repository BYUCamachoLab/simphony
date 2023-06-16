# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.models
===============

This module contains the ``Model`` and ``Subcircuit`` classes. The ``Model``
class is the base class for all models. The ``Subcircuit`` class is where
the subnetwork growth algorithm takes place.

Instances of models are components. As components are connected to each other,
they form a circuit. There are three ways to connect components:

1. ``comp1_or_pin.connect(comp2_or_pin)``
2. ``comp1.multiconnect(comp_or_pin, comp_or_pin, ...)``
3. ``comp1.interface(comp2)``
"""

import os
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as testnp
import jax.numpy as np

# import numpy as np

try:
    from gdsfactory import Component, ComponentReference

    _has_gf = True
except ImportError:
    _has_gf = False

from simphony.connect import create_block_diagonal, innerconnect_s
from simphony.formatters import ModelFormatter, ModelJSONFormatter
from simphony.layout import Circuit
from simphony.pins import Pin, PinList


class Model:
    """The basic element type describing the model for a component with
    scattering parameters.

    Any class that inherits from Model or its subclasses must declare either the
    pin_count or pins attribute. See Attributes for more info.

    Attributes
    ----------
    freq_range : np.ndarray
        A tuple of the valid frequency bounds for the element in the  order
        (lower, upper). Defaults to (-infty, infty).
    pin_count : int
        The number of pins for the device. Must be set if pins is not.
    pins : tuple of str
        A tuple of all the default pin names of the device. Must be set if
        pin_count is not.
    component : Component
        A gdsfactory Component object which is a representation of this component (optional).
    """

    def __init__(
        self,
        name: str = "",
        *,
        freq_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        pins: Optional[List[Pin]] = None,
    ) -> None:
        """Initializes an instance of the model.

        Parameters
        ----------
        name : str
            The name of the model.
        freq_range : tuple of floats
            The frequency range for the model. If not specified,
            it will inherit from cls.freq_range. If that is not specified,
            then it will default to (0, inf).
        pins : list of Pin
            The pins for the model. If not specified, the pins will be
            be initialized from cls.pins. If that is not specified,
            cls.pin_count number of pins will be initialized.
        component : Component
            A gdsfactory Component object for this component (optional).

        Raises
        ------
        NotImplementedError
            when cls.freq_range is undefined.
        NotImplementedError
            when pins is None and cls.pin_count and cls.pins are undefined.
        """

        # Set circuit
        self.circuit = Circuit(self)

    def connect(
        self,
        component_or_pin: Union["Model", Pin],
        component1_ref: "ComponentReference" = None,
        component2_ref: "ComponentReference" = None,
    ) -> "Model":
        """Connects the next available (unconnected) pin from this component to
        the component/pin passed in as the argument.

        If a component is passed in, the first available pin from this
        component is connected to the first available pin from the other
        component.
        """
        if None in (component1_ref, component2_ref):
            self._get_next_unconnected_pin().connect(component_or_pin)
        elif _has_gf:
            self._get_next_unconnected_pin().connect(
                component_or_pin, component1_ref, component2_ref
            )
        else:
            raise ImportError(
                "gdsfactory must be installed to connect gdsfactory components. Try `pip install gdsfactory`."
            )

        return self

    def interface(
        self,
        component: "Model",
        component1_ref: "ComponentReference" = None,
        component2_ref: "ComponentReference" = None,
    ) -> "Model":
        """Interfaces this component to the component passed in by connecting
        pins with the same names.

        Only pins that have been renamed will be connected.
        """
        if None in (component1_ref, component2_ref):
            for selfpin in self.pins:
                for componentpin in component.pins:
                    if selfpin.name[0:3] != "pin" and selfpin.name == componentpin.name:
                        selfpin.connect(componentpin)
        elif _has_gf:
            for selfpin in self.pins:
                for componentpin in component.pins:
                    if selfpin.name[0:3] != "pin" and selfpin.name == componentpin.name:
                        selfpin.connect(componentpin, component1_ref, component2_ref)
        else:
            raise ImportError(
                "gdsfactory must be installed to connect gdsfactory components. Try `pip install gdsfactory`."
            )
        return self

    def monte_carlo_s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Implements the monte carlo routine for the given Model.

        If no monte carlo routine is defined, the default behavior returns the
        result of a call to ``s_parameters()``.

        Parameters
        ----------
        freqs : np.array
            The frequency range to generate monte carlo s-parameters over.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
            Its shape should be (the number of frequency points x ports x ports).
            If the scattering parameters are requested for only a single
            frequency, for example, and the device has 4 ports, the shape
            returned by ``monte_carlo_s_parameters`` would be (1, 4, 4).
        """
        return self.s_parameters(freqs)

    def layout_aware_monte_carlo_s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Implements the monte carlo routine for the given Model.

        If no monte carlo routine is defined, the default behavior returns the
        result of a call to ``s_parameters()``.

        Parameters
        ----------
        freqs : np.array
            The frequency range to generate monte carlo s-parameters over.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
            Its shape should be (the number of frequency points x ports x ports).
            If the scattering parameters are requested for only a single
            frequency, for example, and the device has 4 ports, the shape
            returned by ``monte_carlo_s_parameters`` would be (1, 4, 4).
        """
        return self.s_parameters(freqs)

    def regenerate_monte_carlo_parameters(self) -> None:
        """Regenerates parameters used to generate monte carlo s-matrices.

        If a monte carlo method is not implemented for a given model, this
        method does nothing. However, it can optionally be implemented so that
        parameters are regenerated once per circuit simulation. This ensures
        correlation between all components of the same type that reference
        this model in a circuit. For example, the effective index of a
        waveguide should not be different for each waveguide in a small
        circuit; they will be more or less consistent within a single small
        circuit.

        The ``MonteCarloSweepSimulator`` calls this function once per run over
        the circuit.

        Notes
        -----
        This function should not accept any parameters, but may act on instance
        or class attributes.
        """
        pass

    def regenerate_layout_aware_monte_carlo_parameters(self):
        """Reassigns dimension parameters to the nominal values for the component.

        If a monte carlo method is not implemented for a given model, this
        method does nothing. However, it can optionally be implemented so that
        parameters are reassigned for every run.

        The ``LayoutAwareMonteCarloSweepSimulator`` calls this function once per run per component.

        Notes
        -----
        This function should not accept any parameters, but may act on instance
        or class attributes.
        """
        pass

    def update_variations(self, **kwargs):
        """Update width and thickness variations for the component using correlated
        samples. This is used for layout-aware Monte Carlo runs."""
        pass


class Subcircuit(Model):
    """The ``Subcircuit`` model exposes the ``Model`` API for a group of
    connected components.

    Any unconnected pins from the underlying components are re-exposed.
    This requires that unconnected pins have unique names.

    Parameters
    ----------
    circuit : Circuit
        The circuit to turn into a subcircuit.
    name : str, optional
        An optional name for the subcircuit.
    permanent : bool, optional
        Whether or not this subcircuit should be considered permanent.

        If you intend to make connections to a subcircuit, it is considered
        a permanent subcircuit. Permanet subcircuits require that pin names
        be unique.

    Raises
    ------
    ValueError
        when no components belong to the circuit.
    ValueError
        when unconnected pins share the same name.
    ValueError
        when no unconnected pins exist.
    """

    scache: Dict[Model, "np.ndarray"] = {}

    def __init__(
        self,
        circuit: Circuit,
        name: str = "",
        *,
        permanent: bool = True,
        **kwargs,
    ) -> None:
        freq_range = [0, float("inf")]
        pins = []
        pin_names = {}

        for component in circuit:
            # figure out which pins to re-expose
            for pin in component.pins:
                # re-expose unconnected pins or pins connected to simulators
                if not pin._isconnected(include_simulators=False):
                    if permanent and pin.name in pin_names:
                        raise ValueError(
                            f"Multiple pins named '{pin.name}' cannot exist in a subcircuit."
                        )
                    else:
                        # keep track of the pin to re-expose
                        pins.append(pin)

                        # make the pin's owner this component if permanent
                        if permanent:
                            pin_names[pin.name] = True
                            pin._component = self

        if len(pins) == 0:
            raise ValueError(
                "A subcircuit needs to contain at least one unconnected pin."
            )

        self._wrapped_circuit = circuit

        super().__init__(
            **kwargs,
            freq_range=freq_range,
            name=name,
            pins=pins,
        )

    def __get_s_params_from_cache(
        self, component, freqs: "np.array", s_parameters_method: str = "s_parameters"
    ):
        """Get the s_params from the cache if possible."""
        if s_parameters_method == "s_parameters":
            # each frequency has a different s-matrix, so we need to cache
            # the s-matrices by frequency as well as component

            s_params = []
            for freq in freqs:
                try:
                    # use the cached s-matrix if available
                    # print(self.__class__.scache)
                    s_matrix = self.__class__.scache[component][freq]
                except KeyError:
                    # make sure the frequency dict is created
                    if component not in self.__class__.scache:
                        # print("component not in scache:", component)
                        self.__class__.scache[component] = {}
                    # print("scache:", self.__class__.scache[component])
                    # store the s-matrix for the frequency and component
                    s_matrix = getattr(component, s_parameters_method)(
                        np.array([freq])
                    )[0]
                    # x = tuple(freq)
                    # x = testnp.load(freq, allow_pickle=True)
                    # print("s matrix", s_matrix, "component", component, "freq", freq)
                    # print("freq", type(freq))
                    # print("type x:", type(testnp.array(freq)))
                    x = testnp.array(freq)
                    # print(x)
                    # , freq.at[0].get())
                    x1 = testnp.reshape(x, 1)
                    freqtup = tuple(x1)
                    # print("scache:", self.__class__.scache[component][freqtup])
                    self.__class__.scache[component][freqtup] = s_matrix
                    # print("scache now:", self.__class__.scache[component][freqtup])
                    # self.__class__.scache.at[component][freq].set(s_matrix)

                # add the s-matrix to our list of s-matrices
                s_params.append(s_matrix)

            # convert to numpy array for the rest of the function
            return np.array(s_params)
        elif (
            s_parameters_method == "monte_carlo_s_parameters"
            or "layout_aware_monte_carlo_s_parameters"
        ):
            # don't cache Monte Carlo scattering parameters
            return getattr(component, s_parameters_method)(freqs)

    def __compute_s_block(self, s_block, available_pins, all_pins):
        """Use the subnetwork growth algorithm for each connection."""
        for pin in all_pins:
            # make sure pins only get connected once
            # and pins connected to simulators get skipped
            if (
                pin._isconnected(include_simulators=False)
                and pin in available_pins
                and pin._connection in available_pins
            ):
                # the pin indices in available_pins lines up with the row/column
                # indices in the matrix. as the matrix shrinks, we remove pins
                # from available_pins so the indices always line up
                k = available_pins.index(pin)
                l = available_pins.index(pin._connection)

                s_block = innerconnect_s(s_block, k, l)

                available_pins.remove(pin)
                available_pins.remove(pin._connection)
        return s_block

    def _s_parameters(
        self,
        freqs: "np.array",
        s_parameters_method: str = "s_parameters",
    ) -> "np.ndarray":
        """Returns the scattering parameters for the subcircuit.

        This method will combine the s-matrices of the underlying
        components using the subnetwork growth algorithm.

        Parameters
        ----------
        freqs : np.ndarray
            The list of frequencies to get scattering parameters for.
        s_parameters_method : str, optional
            The method name to call to get the scattering parameters.
            Either 's_parameters' or 'monte_carlo_s_parameters'
        """
        from simphony.simulation import SimulationModel
        from simphony.simulators import Simulator

        all_pins = []
        available_pins = []
        s_block = None

        # merge all of the s_params into one giant block diagonal matrix
        for component in self._wrapped_circuit:
            # simulators don't have scattering parameters
            if isinstance(component, (Simulator, SimulationModel)):
                continue

            # get the s_params from the cache if possible
            s_params = self.__get_s_params_from_cache(
                component, freqs, s_parameters_method
            )

            # merge the s_params into the block diagonal matrix
            if s_block is None:
                if s_params.ndim == 3:
                    s_block = np.stack((np.abs(s_params), np.angle(s_params)), axis=-1)
                else:
                    s_block = s_params
            else:
                s_block = create_block_diagonal(s_block, s_params)

            # keep track of all of the pins (in order) in the circuit
            all_pins += component.pins
            available_pins += component.pins

        # use the subnetwork growth algorithm for each connection
        return self.__compute_s_block(s_block, available_pins, all_pins)

    def monte_carlo_s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the Monte Carlo scattering parameters for the subcircuit."""
        return self._s_parameters(freqs, "monte_carlo_s_parameters")

    def layout_aware_monte_carlo_s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the Monte Carlo scattering parameters for the subcircuit."""
        return self._s_parameters(freqs, "layout_aware_monte_carlo_s_parameters")

    def regenerate_monte_carlo_parameters(self) -> None:
        """Regenerates parameters used to generate Monte Carlo s-matrices."""
        for component in self._wrapped_circuit:
            component.regenerate_monte_carlo_parameters()

    def regenerate_layout_aware_monte_carlo_parameters(self):
        """Reassigns dimension parameters to the nominal values for the component.

        If a monte carlo method is not implemented for a given model, this
        method does nothing. However, it can optionally be implemented so that
        parameters are reassigned for every run.

        The ``LayoutAwareMonteCarloSweepSimulator`` calls this function once per run per component.

        Notes
        -----
        This function should not accept any parameters, but may act on instance
        or class attributes.
        """
        for component in self._wrapped_circuit:
            component.regenerate_layout_aware_monte_carlo_parameters()

    def update_variations(self, **kwargs):
        for component in self._wrapped_circuit:
            component.update_variations(**kwargs)

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the scattering parameters for the subcircuit."""
        return self._s_parameters(freqs)

    @classmethod
    def clear_scache(cls) -> None:
        """Clears the scattering parameters cache."""
        cls.scache = {}
