# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import json
from typing import TYPE_CHECKING, Type

import numpy as np

from simphony.tools import interpolate

if TYPE_CHECKING:
    from simphony import Model


class ModelFormatter:
    """Base model formatter class that is extended to provide functionality for
    converting a component (model instance) to a string and vice-versa."""

    def _to_component(
        self, cls: Type["Model"], freqs: np.array, s_params: np.ndarray
    ) -> "Model":
        """Returns a component that extends from the specified class and is
        defined by the frequencies and scattering parameters provided.

        Parameters
        ----------
        cls :
            The class to extend from (must extend from simphony.Model).
        freqs :
            The list of valid frequencies for the model.
        s_params :
            The scattering parameters for each frequency.
        """

        # create a temporary class that extends from the passed in class
        class Model(cls):
            freq_range = (freqs.min(), freqs.max())
            pin_count = len(s_params[0])

            def s_parameters(self, _freqs: np.array) -> np.ndarray:
                try:
                    return interpolate(_freqs, freqs, s_params)
                except ValueError:
                    raise ValueError(
                        f"Frequencies must be between {freqs.min(), freqs.max()}."
                    )

        return Model()

    def format(self, freqs: np.array, s_params: np.ndarray) -> str:
        """Returns a string representation of the component's scattering
        parameters.

        Parameters
        ----------
        component :
            The component to get a string representation for.
        """
        raise NotImplementedError

    def parse(self, cls: Type["Model"], string: str) -> "Model":
        """Returns a component from the given string.

        Parameters
        ----------
        cls :
            The class type to create an instance of.
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

    def format(self, freqs: np.array, s_params: np.ndarray) -> str:
        return json.dumps({"freqs": freqs, "s_params": s_params}, cls=JSONEncoder)

    def parse(self, cls: Type["Model"], string: str) -> "Model":
        data = json.loads(string, cls=JSONDecoder)
        return self._to_component(
            cls, np.array(data["freqs"]), np.array(data["s_params"])
        )
