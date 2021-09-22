# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.pins
=============

This module contains the logic for managing pins and their connections. When
connections are made, the pins handle letting the components know which in turn
makes sure all components belong to the same ``Circuit`` instance.
"""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from simphony import Model


class Pin:
    """This class represents an individual pin on a component.

    As pins are connected and disconnected from each other, the
    components keep track of which circuit they belong to.
    """

    def __init__(self, component: "Model", name: str) -> None:
        """Instantiates a pin object.

        Parameters
        ----------
        component :
            The component that this pin belongs to.
        name :
            The name of the pin.
        """
        self._component = component
        self._connection = None
        self.name = name

    def _isconnected(self, *, include_simulators: bool = True) -> bool:
        """Returns whether or not this pin is connected to another pin.

        Parameters
        ----------
        include_simulators :
            When true, connections to simulators are counted as connections.
            When false, they are not counted as connections.
        """
        if self._connection is None:
            return False

        if include_simulators:
            return True

        from simphony.simulation import SimulationModel
        from simphony.simulators import Simulator

        return not isinstance(
            self._connection._component, Simulator
        ) and not isinstance(self._connection._component, SimulationModel)

    def connect(self, pin_or_component: Union["Pin", "Model"]) -> None:
        """Connects this pin to the pin/component that is passed in.

        If a component instance is passed in, this pin will connect to
        the first unconnected pin of the component.
        """
        pin = pin_or_component

        # if we are dealing with a component, we want to get it's pin instead
        if not isinstance(pin, Pin):
            pin = pin_or_component._get_next_unconnected_pin()

        # make sure the pin is disconnected before establishing the connection
        self.disconnect()
        pin.disconnect()

        self._connection = pin
        pin._connection = self

        # let the components know that a new connection was established
        self._component._on_connect(pin._component)

    def disconnect(self) -> None:
        """Disconnects this pin to whatever it is connected to."""
        if self._isconnected():
            pin = self._connection

            self._connection = None
            pin._connection = None

            # let the components know that a connection was disconnected
            self._component._on_disconnect(pin._component)

    def rename(self, name: str) -> None:
        """Renames the pin."""
        self.name = name


class PinList(list):
    """Keeps track and manages the pins in a component."""

    def __getitem__(self, index_or_name: Union[int, str]) -> Pin:
        # first try to get the pin by index
        # then try to get the pin by name
        try:
            return super().__getitem__(index_or_name)
        except TypeError:
            for pin in self:
                if pin.name == index_or_name:
                    return pin

            raise IndexError(f"Pin '{index_or_name}' does not exist.")

    def __init__(
        self, component_or_pins: Union[list, "Model"], length: int = 0
    ) -> None:
        """Initializes the pin list. You can pass in either a component with a
        length, or a list of ``Pin`` objects to instantiate the ``PinList``.

        Parameters
        ----------
        component_or_pins :
            Either the component that the pins belong to or a list of ``Pin``
            objects.
        length :
            The number of pins that the component has.
        """
        if length:
            super().__init__(
                [Pin(component_or_pins, f"pin{i + 1}") for i in range(length)]
            )
        else:
            super().__init__(component_or_pins)

    def rename(self, *names: str) -> None:
        """Renames the pins for this pinlist.

        The first pin is renamed to the first value passed in, the
        second pin is renamed to the second value, etc.
        """
        try:
            for index, name in enumerate(names):
                self.__getitem__(index).rename(name)
        except IndexError:
            raise ValueError(f"Pin {index + 1} does not exist.")
