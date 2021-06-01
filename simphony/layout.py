# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from typing import TYPE_CHECKING, Type

from simphony.formatters import CircuitFormatter, CircuitJSONFormatter

if TYPE_CHECKING:
    import numpy as np

    from simphony import Model
    from simphony.models import Subcircuit


class Circuit(list):
    """The ``Circuit`` class keeps an ordered list of components.

    The components themselves manage which circuits they belong to as
    they are connected and disconnected from one another.
    """

    def __init__(self, component: "Model") -> None:
        """Initializes a circuit.

        Parameters
        ----------
        component :
            The first component of a circuit
        """
        super().__init__([component])

    def __str__(self) -> str:
        return self._str_recursive().rstrip()

    def _add(self, component: "Model") -> bool:
        """Adds the specified component to the circuit if it isn't already
        included.

        Returns whether or not the component was added.
        """
        if component not in self:
            self.append(component)
            return True

        return False

    def _merge(self, other: "Circuit") -> None:
        """Merges the other circuit into this circuit.

        All components in the other circuit have their circuit reference
        updated accordingly.
        """
        for component in other:
            if self._add(component):
                component.circuit = self

    def _str_recursive(self, indent: int = 0) -> str:
        """A recursive function to generate a string representation of the
        circuit."""
        result = ""
        for component in self:
            result += f"{' ' * indent * 2} {component}\n"

            if hasattr(component, "_wrapped_circuit"):
                indent += 1
                result += component._wrapped_circuit._str_recursive(indent)
                indent -= 1

        return result

    def s_parameters(self, freqs: "np.ndarray") -> "np.ndarray":
        """Returns the scattering parameters for the circuit."""
        return self.to_subcircuit(permanent=False).s_parameters(freqs)

    def to_file(
        self,
        filename: str,
        freqs: "np.array",
        *,
        Formatter: Type[CircuitFormatter] = CircuitJSONFormatter,
    ) -> None:
        """Writes a string representation of this circuit to a file.

        Parameters
        ----------
        filename :
            The name of the file to write to.
        freqs :
            The list of frequencies to save data for.
        Formatter :
            The class of the formatter to use.
        """
        with open(filename, "w") as file:
            file.write(Formatter().format(self, freqs))
            file.close()

    def to_subcircuit(self, name: str = "", **kwargs) -> "Subcircuit":
        """Converts this circuit into a subcircuit component for easy re-use in
        another circuit."""
        from simphony.models import Subcircuit

        return Subcircuit(self, **kwargs, name=name)

    @staticmethod
    def from_file(
        filename: str, *, Formatter: Type[CircuitFormatter] = CircuitJSONFormatter
    ) -> "Circuit":
        """Creates a circuit from a file using the specified formatter.

        Parameters
        ----------
        filename :
            The filename to read from.
        Formatter :
            The class of the formatter to use.
        """
        with open(filename, "r") as file:
            circuit = Formatter().parse(file.read())
            file.close()

        return circuit
