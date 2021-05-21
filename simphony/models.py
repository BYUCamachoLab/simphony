# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Union

from simphony.layout import Circuit
from simphony.pins import Pin, PinList

if TYPE_CHECKING:
    from numpy import ndarray


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
            then it will default to (-infty, infty).
        pins :
            The pins for the model. If not specified, the pins will be
            be initialized from cls.pins. If that is not specified,
            cls.pin_count number of pins will be initialized. If pins is not
            passed in, cls.pin_count or cls.pins must be defined.

        Raises
        ------
        NotImplementedError
            when cls.pin_count and cls.pins are both undefined.
        """
        self.circuit = Circuit(self)

        # set the frequency range for the instnace. resolution order:
        # 1. freq_range being passed in to __init__
        # 2. cls.freq_range
        # 3. default value
        if freq_range:
            self.freq_range = freq_range
        else:
            try:
                self.freq_range = self.__class__.freq_range
            except AttributeError:
                self.freq_range = (None, None)

        self.name = name

        # initiate the Pin objects for the instance. resolution order:
        # 1. list of pins being passed in to __init__
        # 2. cls.pins initializes and renames pins
        # 3. cls.pin_count initializes pins
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

        if circuit1 != circuit2:
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

    def connect(self, component_or_pin: Union["Model", Pin]) -> None:
        """Connects the next available (unconnected) pin from this component to
        the component/pin passed in as the argument.

        If a component is passed in, the first available pin from this
        component is connected to the first available pin from the other
        component.
        """
        self._get_next_unconnected_pin().connect(component_or_pin)

    def interface(self, component: "Model") -> None:
        """Interfaces this component to the component passed in by connecting
        pins with the same names."""
        for selfpin in self.pins:
            for componentpin in component.pins:
                if selfpin.name == componentpin.name:
                    selfpin.connect(componentpin)

    def multiconnect(self, *connections: Union["Model", Pin]) -> None:
        """Connects this component to the specified connections by looping
        through each connection and connecting it with the next available
        (unconnected) pin from this component.

        See the ``connect`` method for more information.
        """
        for connection in connections:
            self.connect(connection)

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

    def s_parameters(self, freq: "ndarray") -> "ndarray":
        """Returns scattering parameters for the element with its given
        parameters as declared in the optional ``__init__()``.

        Parameters
        ----------
        freq : np.ndarray
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

    # TODO
    # def to_file(self, filename: str, *, formatter=None):
    #     pass

    # @classmethod
    # def from_file(cls, filename: str, *, formatter=None):
    #     pass


class Subcircuit(Model):
    """The ``Subcircuit`` model exposes the ``Model`` API for a group of
    connected components.

    Any unconnected pins from the underlying components are re-exposed.
    This requires that unconnected pins have unique names.
    """

    def __init__(
        self, circuit: Circuit, name: str = "", *, rename_pins: bool = False, **kwargs
    ) -> None:
        """Initializes a subcircuit from the given circuit.

        Parameters
        ----------
        circuit :
            The circuit to turn into a subcircuit.
        name :
            An optional name for the subcircuit.
        rename_pins :
            Whether or not to rename the re-exposed unconnected pins to prevent
            naming collisions.

        Raises
        ------
        ValueError
            when no components belong to the circuit.
        ValueError
            when unconnected pins share the same name.
        ValueError
            when no unconnected pins exist.
        """
        if len(circuit) == 0:
            raise ValueError(
                "A circuit needs to contain at least one component to be a subcircuit."
            )

        pins = []
        pin_names = {}
        pin_number = 1

        # figure out which pins to re-expose
        for component in circuit:
            for pin in component.pins:
                if not pin._isconnected():
                    if rename_pins:
                        pin.rename(f"pin{pin_number}")
                        pin_number += 1

                    if pin.name in pin_names:
                        raise ValueError(
                            f"Multiple pins named '{pin.name}' cannot exist in a subcircuit."
                        )
                    else:
                        # keep track of the pin to re-expose
                        # and set the subcircuit as the pin's component
                        pins.append(pin)
                        pin._component = self
                        pin_names[pin.name] = True

        if len(pins) == 0:
            raise ValueError(
                "A subcircuit needs to contain at least one unconnected pin."
            )

        self._wrapped_circuit = circuit

        super().__init__(**kwargs, name=name, pins=pins)

    def s_parameters(self, freq: "ndarray") -> "ndarray":
        """Returns the scattering parameters for the subcircuit.

        This method will combine the s-matrices of the underlying
        components using the subnetwork growth algorithm.
        """
        for component in self._wrapped_circuit:
            # TODO: combine s_parameters from components into one
            pass
