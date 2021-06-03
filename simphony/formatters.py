# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import json
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from simphony.tools import interpolate

if TYPE_CHECKING:
    from simphony import Model
    from simphony.layout import Circuit


class ModelFormatter:
    """Base model formatter class that is extended to provide functionality for
    converting a component (model instance) to a string and vice-versa."""

    flatten_subcircuits = False

    def _from_component(
        self, component: "Model", freqs: np.array
    ) -> Tuple[str, List[str], Optional[np.ndarray], Optional[str]]:
        """Gets the component's information that needs to be formatted.

        Parameters
        ----------
        component :
            The component to get the information from.
        freqs :
            The list of frequencies to get information about.
        """
        name = component.name or f"{component.__class__.__name__} component"
        pins = [pin.name for pin in component.pins]

        # if the component is a subcircuit, save the underyling circuit unless
        # we have been asked to flatten it
        if hasattr(component, "_wrapped_circuit") and not self.flatten_subcircuits:
            s_params = None
            subcircuit = CircuitJSONFormatter().format(
                component._wrapped_circuit, freqs
            )
        else:
            s_params = component.s_parameters(freqs)
            subcircuit = None

        return (name, pins, s_params, subcircuit)

    def _to_component(
        self,
        freqs: np.array,
        name: str,
        pins: List[str],
        s_params: Optional[np.ndarray] = None,
        subcircuit: Optional[str] = None,
    ) -> "Model":
        """Returns a component that is defined by the given parameters.

        If the component is a subcircuit, s_params will be None and subcircuit
        will not be None. Otherwise, s_params will be populated and subcircuit
        will be None.

        Parameters
        ----------
        freqs :
            The list of valid frequencies for the model.
        name :
            The name of the component.
        pins :
            The pins names for the component.
        s_params :
            The scattering parameters for each frequency.
        subcircuit :
            If the component is a subcircuit, this contains the circuit information.
        """
        from simphony.models import Model, Subcircuit

        if subcircuit is not None:
            # instantiate a subcircuit if there is subcircuit information
            component = Subcircuit(CircuitJSONFormatter().parse(subcircuit))
        else:
            # instantiate a static model instance if s_params is given
            class StaticModel(Model):
                freq_range = (freqs.min(), freqs.max())
                pin_count = len(s_params[0])

                def s_parameters(self, _freqs: np.array) -> np.ndarray:
                    try:
                        return interpolate(_freqs, freqs, s_params)
                    except ValueError:
                        raise ValueError(
                            f"Frequencies must be between {freqs.min(), freqs.max()}."
                        )

            component = StaticModel()

        component.name = name
        component.rename_pins(*pins)

        return component

    def format(self, component: "Model", freqs: np.array) -> str:
        """Returns a string representation of the component's scattering
        parameters.

        Parameters
        ----------
        component :
            The component to format.
        freqs :
            The frequencies to get scattering parameters for.
        """
        raise NotImplementedError

    def parse(self, string: str) -> "Model":
        """Returns a component from the given string.

        Parameters
        ----------
        string :
            The string to parse.
        """
        raise NotImplementedError


class JSONEncoder(json.JSONEncoder):
    """JSON Encoder class that handles np.ndarray and complex object types."""

    def default(self, object):
        # the default method is called for each object.
        # if it's an ndarray or complex object, we encode it
        # otherwise, we use the default encoder
        if isinstance(object, np.ndarray):
            return object.tolist()
        elif isinstance(object, complex):
            return {"r": object.real, "i": object.imag}
        else:
            return super().default(object)


class JSONDecoder(json.JSONDecoder):
    """JSON Decoder class that handles complex object types."""

    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def object_hook(self, dict):
        # the object_hook method gets called whenever an object is found
        # if the object represents a complex number, we decode to that
        return complex(dict["r"], dict["i"]) if "r" in dict else dict


class ModelJSONFormatter(ModelFormatter):
    """The ModelJSONFormatter class formats the model data in a JSON format."""

    def format(self, component: "Model", freqs: np.array) -> str:
        name, pins, s_params, subcircuit = self._from_component(component, freqs)
        return json.dumps(
            {
                "freqs": freqs,
                "name": name,
                "pins": pins,
                "s_params": s_params,
                "subcircuit": subcircuit,
            },
            cls=JSONEncoder,
        )

    def parse(self, string: str) -> "Model":
        data = json.loads(string, cls=JSONDecoder)
        return self._to_component(
            np.array(data["freqs"]),
            data["name"],
            data["pins"],
            np.array(data["s_params"]),
            data["subcircuit"],
        )


class CircuitFormatter:
    """Base circuit formatter class that is extended to provide functionality
    for converting a circuit to a string and vice-versa."""

    def format(self, circuit: "Circuit", freqs: np.array) -> str:
        """Returns a string representation of the circuit.

        Parameters
        ----------
        circuit :
            The circuit to get a string representation for.
        """
        raise NotImplementedError

    def parse(self, string: str) -> "Circuit":
        """Returns a circuit from the given string.

        Parameters
        ----------
        string :
            The string to parse.
        """
        raise NotImplementedError


class CircuitJSONFormatter:
    """This class handles converting a circuit to JSON and vice-versa."""

    def format(self, circuit: "Circuit", freqs: np.array) -> str:
        from simphony.simulators import Simulator

        data = {"components": [], "connections": []}
        for i, component in enumerate(circuit):
            # skip simulators
            if isinstance(component, Simulator):
                continue

            # get a representation for each component
            data["components"].append(
                component.to_string(freqs, Formatter=ModelJSONFormatter)
            )

            # get all of the connections between components
            for j, pin in enumerate(component.pins):
                if pin._isconnected(include_simulators=False):
                    try:
                        # we only care about saving connections within this
                        # circuit. if the index does not exist, just ignore it
                        k = circuit.index(pin._connection._component)
                        l = circuit[k].pins.index(pin._connection)

                        # only store connections one time
                        if i < k or (i == k and j < l):
                            data["connections"].append((i, j, k, l))
                    except ValueError:
                        pass

        return json.dumps(data)

    def parse(self, string: str) -> "Circuit":
        from simphony import Model

        data = json.loads(string)

        # load all of the components
        components = []
        for string in data["components"]:
            components.append(Model.from_string(string, Formatter=ModelJSONFormatter))

        # connect the components to each other
        for i, j, k, l in data["connections"]:
            components[i].pins[j].connect(components[k].pins[l])

        return components[0].circuit
