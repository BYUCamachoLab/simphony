# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import json
from typing import TYPE_CHECKING

import numpy as np

from simphony.tools import interpolate

if TYPE_CHECKING:
    from simphony import Model
    from simphony.layout import Circuit


class ModelFormatter:
    """Base model formatter class that is extended to provide functionality for
    converting a component (model instance) to a string and vice-versa."""

    def _to_component(
        self, name: str, freqs: np.array, s_params: np.ndarray
    ) -> "Model":
        """Returns a component that is defined by the frequencies and
        scattering parameters provided.

        Parameters
        ----------
        name :
            The name of the component.
        freqs :
            The list of valid frequencies for the model.
        s_params :
            The scattering parameters for each frequency.
        """
        from simphony import Model

        # create a temporary class that extends from the passed in class
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

        return component

    def format(self, name: str, freqs: np.array, s_params: np.ndarray) -> str:
        """Returns a string representation of the component's scattering
        parameters.

        Parameters
        ----------
        name :
            The name of the component that is being formatted.
        freqs :
            The frequencies to get scattering parameters for.
        s_params :
            The scattering parameters for the corresponding frequencies.
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

    def format(self, name: str, freqs: np.array, s_params: np.ndarray) -> str:
        return json.dumps(
            {"freqs": freqs, "name": name, "s_params": s_params}, cls=JSONEncoder
        )

    def parse(self, string: str) -> "Model":
        data = json.loads(string, cls=JSONDecoder)
        return self._to_component(
            data["name"], np.array(data["freqs"]), np.array(data["s_params"])
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
    """This class handles converting circuits to JSON and JSON to
    components."""

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
                    k = circuit.index(pin._connection._component)
                    l = circuit[k].pins.index(pin._connection)

                    # only store connections one time
                    if i < k or (i == k and j < l):
                        data["connections"].append((i, j, k, l))

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
