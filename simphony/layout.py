# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.layout
===============

This module contains the ``Circuit`` object. The ``Circuit`` object acts as
a sorted set that contains components. As components connect/disconnect to each
other, they will make sure that they belong to the same ``Circuit`` instance.
"""

import os
from typing import TYPE_CHECKING, List, Optional

from simphony.formatters import CircuitFormatter, CircuitJSONFormatter

if TYPE_CHECKING:
    import numpy as np

    from simphony import Model
    from simphony.models import Subcircuit
    from simphony.pins import Pin


class Circuit(list):
    """The ``Circuit`` class keeps an ordered list of components.

    The components themselves manage which circuits they belong to as
    they are connected and disconnected from one another.
    """

    def __hash__(self) -> int:
        """Gets a hash for the circuit based on components and connections."""
        from simphony.simulators import Simulator

        components = 0
        connections = 0
        for component in self:
            if not isinstance(component, Simulator):
                components += hash(component)
                for pin in component.pins:
                    if pin._isconnected(include_simulators=False):
                        connections += hash(hash(pin) + hash(pin._connection))

        return hash(components + connections)

    def __init__(self, component: "Model") -> None:
        """Initializes a circuit.

        Parameters
        ----------
        component :
            The first component of a circuit
        """
        super().__init__([component])

    def __str__(self) -> str:
        return self._str_recursive(components=self._get_components()).rstrip()

    def _add(self, component: "Model") -> bool:
        """Adds the specified component to the circuit if it isn't already
        included.

        Returns whether or not the component was added.
        """
        if component not in self:
            self.append(component)
            return True

        return False

    def _get_components(self) -> List["Model"]:
        """Gets a list of all components contained in this circuit.

        This includes any components in any subcircuits.
        """
        components = []
        for component in self:
            components.append(component)
            if hasattr(component, "_wrapped_circuit"):
                components += component._wrapped_circuit._get_components()

        return components

    def _merge(self, other: "Circuit") -> None:
        """Merges the other circuit into this circuit.

        All components in the other circuit have their circuit reference
        updated accordingly.
        """
        for component in other:
            if self._add(component):
                component.circuit = self

    def _str_recursive(self, indent: int = 0, components: List["Model"] = []) -> str:
        """A recursive function to generate a string representation of the
        circuit."""
        spacing = 3
        output = ""
        for component in self:
            # add this component to the output
            output += f"{' ' * indent * spacing}"
            output += f"[{components.index(component)}] {component}\n"

            # if the component is a subcircuit, recurse into it
            if hasattr(component, "_wrapped_circuit"):
                indent += 1
                output += component._wrapped_circuit._str_recursive(indent, components)
                indent -= 1
            else:
                # list all of the pin information
                for pin in component.pins:
                    output += f"{' ' * (indent + 1) * spacing}"

                    # indicate when multiple components reference the same pin
                    # this happens with subcircuit components
                    if pin._component != component:
                        output += "*"

                    # add the pin info to the output
                    output += f"[{components.index(pin._component)}][{pin.name}]"

                    # if the pin is connected, add the connection info
                    if pin._isconnected():
                        i = components.index(pin._connection._component)
                        output += f" - [{i}][{pin._connection.name}]"

                    output += "\n"

        return output

    @property
    def pins(self) -> List["Pin"]:
        """Returns the pins for the circuit."""
        return self.to_subcircuit(permanent=False).pins

    def get_pin_index(self, pin: "Pin") -> int:
        """Gets the pin index for the specified pin in the scattering
        parameters."""
        for i, _pin in enumerate(self.pins):
            if _pin == pin:
                return i

        raise ValueError("The pin must belong to the circuit.")

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        """Returns the scattering parameters for the circuit."""
        return self.to_subcircuit(permanent=False).s_parameters(freqs)

    def to_file(
        self,
        filename: str,
        freqs: "np.array",
        *,
        formatter: Optional[CircuitFormatter] = None,
    ) -> None:
        """Writes a string representation of this circuit to a file.

        Parameters
        ----------
        filename :
            The name of the file to write to.
        freqs :
            The list of frequencies to save data for.
        formatter :
            The formatter instance to use.
        """
        formatter = formatter if formatter is not None else CircuitJSONFormatter()

        # change the cwd to the the directory containing the file
        filename = os.path.abspath(filename)
        cwd = os.getcwd()
        dir, _ = os.path.split(filename)
        os.chdir(dir)

        # format the file
        with open(filename, "w") as file:
            file.write(formatter.format(self, freqs))
            file.close()

        # restore the cwd
        os.chdir(cwd)

    def to_subcircuit(self, name: str = "", **kwargs) -> "Subcircuit":
        """Converts this circuit into a subcircuit component for easy re-use in
        another circuit."""
        from simphony.models import Subcircuit

        return Subcircuit(self, **kwargs, name=name)

    @staticmethod
    def from_file(
        filename: str, *, formatter: Optional[CircuitFormatter] = None
    ) -> "Circuit":
        """Creates a circuit from a file using the specified formatter.

        Parameters
        ----------
        filename :
            The filename to read from.
        formatter :
            The formatter instance to use.
        """
        formatter = formatter if formatter is not None else CircuitJSONFormatter()

        # change the cwd to the the directory containing the file
        filename = os.path.abspath(filename)
        cwd = os.getcwd()
        dir, _ = os.path.split(filename)
        os.chdir(dir)

        # parse the file
        with open(filename, "r") as file:
            circuit = formatter.parse(file.read())
            file.close()

        # restore the cwd
        os.chdir(cwd)

        return circuit
