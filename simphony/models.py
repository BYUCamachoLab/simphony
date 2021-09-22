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

import numpy as np

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
    freq_range :
        A tuple of the valid frequency bounds for the element in the  order
        (lower, upper). Defaults to (-infty, infty).
    pin_count :
        The number of pins for the device. Must be set if pins is not.
    pins :
        A tuple of all the default pin names of the device. Must be set if
        pin_count is not.
    """

    freq_range: ClassVar[Tuple[Optional[float], Optional[float]]]
    pin_count: ClassVar[Optional[int]]
    pins: ClassVar[Optional[Tuple[str, ...]]]
    pins: PinList  # additional type hint for instance.pins

    def __getitem__(self, item: Union[int, str]) -> Pin:
        return self.pins[item]

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
        name :
            The name of the model.
        freq_range :
            The frequency range for the model. If not specified,
            it will inherit from cls.freq_range. If that is not specified,
            then it will default to (0, inf).
        pins :
            The pins for the model. If not specified, the pins will be
            be initialized from cls.pins. If that is not specified,
            cls.pin_count number of pins will be initialized.

        Raises
        ------
        NotImplementedError
            when cls.freq_range is undefined.
        NotImplementedError
            when pins is None and cls.pin_count and cls.pins are undefined.
        """
        self.circuit = Circuit(self)

        # set the frequency range for the instance. resolution order:
        # 1. freq_range
        # 2. cls.freq_range
        # 3. default value (0, inf)
        if freq_range:
            self.freq_range = (freq_range[0], freq_range[1])
        else:
            try:
                self.freq_range = self.__class__.freq_range
            except AttributeError:
                self.freq_range = (0, float("inf"))

        self.name = name

        # initiate the Pin objects for the instance. resolution order:
        # 1. pins (list of Pin objects)
        # 2. cls.pins (tuple of pin names)
        # 3. cls.pin_count
        if pins:
            self.pins = PinList(pins)
        else:
            try:
                self.pins = PinList(self, len(self.__class__.pins))
                self.pins.rename(*self.__class__.pins)
            except AttributeError:
                try:
                    self.pins = PinList(self, self.__class__.pin_count)
                except AttributeError:
                    name = self.__class__.__name__
                    raise NotImplementedError(
                        f"{name}.pin_count or {name}.pins needs to be defined."
                    )

    def __str__(self) -> str:
        name = self.name or f"{self.__class__.__name__} component"
        return f"{name} with pins: {', '.join([pin.name for pin in self.pins])}"

    def _get_next_unconnected_pin(self) -> Pin:
        """Loops through this model's pins and returns the next unconnected
        pin.

        Raises
        ------
        ValueError
            when this instance has no unconnected pins.
        """
        for pin in self.pins:
            if not pin._isconnected():
                return pin

        raise ValueError(f"{self.__class__.__name__} has no unconnected pins.")

    def _isconnected(self) -> bool:
        """Returns whether this component is connected to other components."""
        for pin in self.pins:
            if pin._isconnected():
                return True

        return False

    def _on_connect(self, component: "Model") -> None:
        """This method is called whenever one of this component's pins is
        connected to another component.

        This method makes sure that all connected components belong to
        the same circuit. i.e. Their .circuit references all point to
        the same thing.
        """
        if self.circuit != component.circuit:
            # make sure to merge the smaller circuit into the larger
            if len(component.circuit) > len(self.circuit):
                component.circuit._merge(self.circuit)
            else:
                self.circuit._merge(component.circuit)

    def _on_disconnect(self, component: "Model") -> None:
        """This method is called whenever one of this component's pins is
        disconnected to another component.

        This method makes sure that all connected components belong to
        the same circuit. i.e. Their .circuit references all point to
        the same thing.
        """
        circuit1 = Circuit(self)
        circuit2 = Circuit(component)

        # the recursive functions loops through all connected components
        # and adds them to the respective circuits
        self._on_disconnect_recursive(circuit1)
        component._on_disconnect_recursive(circuit2)

        # compare the components in the circuits to see if they're different
        different = len(circuit1) != len(circuit2)
        if not different:
            for component in circuit1:
                if component not in circuit2:
                    different = True
                    break

        if different:
            # we have two separate circuits, but the recursive construction of
            # the circuits destroys the component ordering, so we need to
            # reconstruct the separate circuits in a way that preserves order
            ordered1 = []
            ordered2 = []

            # self.circuit still has all of the components in order, so we will
            # loop through them and sort them into two lists
            for component in self.circuit:
                if component in circuit1:
                    ordered1.append(component)
                else:
                    ordered2.append(component)

            # now we create the two separate circuits, add the ordered
            # components to them, and make the components point to the circuits
            circuit1 = Circuit(ordered1[0])
            circuit2 = Circuit(ordered2[0])

            for component in ordered1:
                circuit1._add(component)
                component.circuit = circuit1

            for component in ordered2:
                circuit2._add(component)
                component.circuit = circuit2

    def _on_disconnect_recursive(self, circuit: Circuit) -> None:
        """Recursive logic for ``_on_disconnect``.

        It loops through all of the pins for this component and makes
        sure that the connected components are all a part of the
        circuit. It then proceeds to loop through the pins for the
        connected components. Once all components have been visited, the
        recursion ends.
        """
        for pin in self.pins:
            if pin._isconnected():
                component = pin._connection._component
                if circuit._add(component):
                    component._on_disconnect_recursive(circuit)

    def connect(self, component_or_pin: Union["Model", Pin]) -> "Model":
        """Connects the next available (unconnected) pin from this component to
        the component/pin passed in as the argument.

        If a component is passed in, the first available pin from this
        component is connected to the first available pin from the other
        component.
        """
        self._get_next_unconnected_pin().connect(component_or_pin)
        return self

    def disconnect(self) -> None:
        """Disconnects this component from all other components."""
        for pin in self.pins:
            pin.disconnect()

    def interface(self, component: "Model") -> "Model":
        """Interfaces this component to the component passed in by connecting
        pins with the same names.

        Only pins that have been renamed will be connected.
        """
        for selfpin in self.pins:
            for componentpin in component.pins:
                if selfpin.name[0:3] != "pin" and selfpin.name == componentpin.name:
                    selfpin.connect(componentpin)

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

    def multiconnect(self, *connections: Union["Model", Pin, None]) -> "Model":
        """Connects this component to the specified connections by looping
        through each connection and connecting it with the corresponding pin.

        The first connection is connected to the first pin, the second
        connection to the second pin, etc. If the connection is set to None,
        that pin is skipped.

        See the ``connect`` method for more information if the connection is
        a component or a pin.
        """
        for index, connection in enumerate(connections):
            if connection is not None:
                self.pins[index].connect(connection)

        return self

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

        The ``MonteCarloSweepSimulation`` calls this function once per run over
        the circuit.

        Notes
        -----
        This function should not accept any parameters, but may act on instance
        or class attributes.
        """
        pass

    def rename_pins(self, *names: str) -> None:
        """Renames the pins for this component.

        The first pin is renamed to the first value passed in, the
        second pin is renamed to the second value, etc.
        """
        self.pins.rename(*names)

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns scattering parameters for the element with its given
        parameters as declared in the optional ``__init__()``.

        Parameters
        ----------
        freqs : np.array
            The frequency range to get scattering parameters for.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
            Its shape should be (the number of frequency points x ports x ports).
            If the scattering parameters are requested for only a single
            frequency, for example, and the device has 4 ports, the shape
            returned by ``s_parameters`` would be (1, 4, 4).

        Raises
        ------
        NotImplementedError
            Raised if the subclassing element doesn't implement this function.
        """
        raise NotImplementedError

    def to_file(
        self,
        filename: str,
        freqs: "np.array",
        *,
        formatter: Optional[ModelFormatter] = None,
    ) -> None:
        """Writes this component's scattering parameters to the specified file
        using the specified formatter.

        Parameters
        ----------
        filename :
            The name of the file to write to.
        freqs :
            The list of frequencies to save data for.
        formatter :
            The formatter instance to use.
        """
        # change the cwd to the the directory containing the file
        filename = os.path.abspath(filename)
        cwd = os.getcwd()
        dir, _ = os.path.split(filename)
        os.chdir(dir)

        # format the file
        with open(filename, "w") as file:
            file.write(self.to_string(freqs, formatter=formatter))
            file.close()

        # restore the cwd
        os.chdir(cwd)

    def to_string(
        self, freqs: "np.array", *, formatter: Optional[ModelFormatter] = None
    ) -> str:
        """Returns this component's scattering parameters as a formatted
        string.

        Parameters
        ----------
        freqs :
            The list of frequencies to save data for.
        formatter :
            The formatter instance to use.
        """
        formatter = formatter if formatter is not None else ModelJSONFormatter()
        return formatter.format(self, freqs)

    @staticmethod
    def from_file(
        filename: str, *, formatter: Optional[ModelFormatter] = None
    ) -> "Model":
        """Creates a component from a file using the specified formatter.

        Parameters
        ----------
        filename :
            The filename to read from.
        formatter :
            The formatter instance to use.
        """
        # change the cwd to the the directory containing the file
        filename = os.path.abspath(filename)
        cwd = os.getcwd()
        dir, _ = os.path.split(filename)
        os.chdir(dir)

        # parse the file
        with open(filename, "r") as file:
            component = Model.from_string(file.read(), formatter=formatter)
            file.close()

        # restore the cwd
        os.chdir(cwd)

        return component

    @staticmethod
    def from_string(
        string: str, *, formatter: Optional[ModelFormatter] = None
    ) -> "Model":
        """Creates a component from a string using the specified formatter.

        Parameters
        ----------
        string :
            The string to load the component from.
        formatter :
            The formatter instance to use.
        """
        formatter = formatter if formatter is not None else ModelJSONFormatter()
        return formatter.parse(string)


class Subcircuit(Model):
    """The ``Subcircuit`` model exposes the ``Model`` API for a group of
    connected components.

    Any unconnected pins from the underlying components are re-exposed.
    This requires that unconnected pins have unique names.
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
        """Initializes a subcircuit from the given circuit.

        Parameters
        ----------
        circuit :
            The circuit to turn into a subcircuit.
        name :
            An optional name for the subcircuit.
        permanent :
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
        freq_range = [0, float("inf")]
        pins = []
        pin_names = {}

        for component in circuit:
            # calculate the frequency range for the subcircuit
            if component.freq_range[0] > freq_range[0]:
                freq_range[0] = component.freq_range[0]
            if component.freq_range[1] < freq_range[1]:
                freq_range[1] = component.freq_range[1]

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

        super().__init__(**kwargs, freq_range=freq_range, name=name, pins=pins)

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
        freqs :
            The list of frequencies to get scattering parameters for.
        s_parameters_method :
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
            if isinstance(component, Simulator) or isinstance(
                component, SimulationModel
            ):
                continue

            # get the s_params from the cache if possible
            if s_parameters_method == "s_parameters":
                # each frequency has a different s-matrix, so we need to cache
                # the s-matrices by frequency as well as component
                s_params = []
                for freq in freqs:
                    try:
                        # use the cached s-matrix if available
                        s_matrix = self.__class__.scache[component][freq]
                    except KeyError:
                        # make sure the frequency dict is created
                        if component not in self.__class__.scache:
                            self.__class__.scache[component] = {}

                        # store the s-matrix for the frequency and component
                        s_matrix = getattr(component, s_parameters_method)(
                            np.array([freq])
                        )[0]
                        self.__class__.scache[component][freq] = s_matrix

                    # add the s-matrix to our list of s-matrices
                    s_params.append(s_matrix)

                # convert to numpy array for the rest of the function
                s_params = np.array(s_params)
            elif s_parameters_method == "monte_carlo_s_parameters":
                # don't cache Monte Carlo scattering parameters
                s_params = getattr(component, s_parameters_method)(freqs)

            # merge the s_params into the block diagonal matrix
            if s_block is None:
                s_block = s_params
            else:
                s_block = create_block_diagonal(s_block, s_params)

            # keep track of all of the pins (in order) in the circuit
            all_pins += component.pins
            available_pins += component.pins

        # use the subnetwork growth algorithm for each connection
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

    def monte_carlo_s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the Monte Carlo scattering parameters for the subcircuit."""
        return self._s_parameters(freqs, "monte_carlo_s_parameters")

    def regenerate_monte_carlo_parameters(self) -> None:
        """Regenerates parameters used to generate Monte Carlo s-matrices."""
        for component in self._wrapped_circuit:
            component.regenerate_monte_carlo_parameters()

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the scattering parameters for the subcircuit."""
        return self._s_parameters(freqs)

    @classmethod
    def clear_scache(cls) -> None:
        """Clears the scattering parameters cache."""
        cls.scache = {}
