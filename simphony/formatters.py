# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.formatters
===================

This module contains two types of classes: ``ModelFormatters`` and
``CircuitFormatters``. These classes are used to serialize / unseralize models
and circuits.

Specifically, instances of these classes should be used in the ``to_file`` and
``from_file`` methods on ``Model`` and ``Circuit``.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
        from simphony.simulation import SimulationModel
        from simphony.simulators import Simulator

        data = {"components": [], "connections": []}
        for i, component in enumerate(circuit):
            # skip simulators
            if isinstance(component, Simulator) or isinstance(
                component, SimulationModel
            ):
                continue

            # get a representation for each component
            data["components"].append(
                component.to_string(freqs, formatter=ModelJSONFormatter())
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
            components.append(Model.from_string(string, formatter=ModelJSONFormatter()))

        # connect the components to each other
        for i, j, k, l in data["connections"]:
            components[i].pins[j].connect(components[k].pins[l])

        return components[0].circuit


class CircuitSiEPICFormatter(CircuitFormatter):
    """This class saves/loads circuits in the SiEPIC SPICE format."""

    mappings = {
        "simphony.libraries.siepic": {
            "ebeam_bdc_te1550": {"name": "BidirectionalCoupler", "parameters": {}},
            "ebeam_dc_halfring_straight": {"name": "HalfRing", "parameters": {}},
            "ebeam_dc_te1550": {"name": "DirectionalCoupler", "parameters": {}},
            "ebeam_gc_te1550": {"name": "GratingCoupler", "parameters": {}},
            "ebeam_terminator_te1550": {"name": "Terminator", "parameters": {}},
            "ebeam_wg_integral_1550": {
                "name": "Waveguide",
                "parameters": {
                    "wg_length": "length",
                    "wg_width": "width",
                },
            },
            "ebeam_y_1550": {"name": "YBranch", "parameters": {}},
        },
    }

    def __init__(self, pdk=None) -> None:
        """Initializes a new formatter.

        Parameters
        ----------
        pdk :
            The PDK to use to instantiate components. Defaults to SiEPIC.
        """
        from simphony.libraries import siepic

        self.pdk = pdk if pdk is not None else siepic

    def _instantiate_component(self, component: Dict[str, Any]) -> "Model":
        """Instantiates a component from the given component data."""
        # get the mapping information
        mapping = self.__class__.mappings[self.pdk.__name__][component["model"]]

        # remap the parameter values so they match the components' API
        parameters = {}
        for k, v in component["params"].items():
            if k in mapping["parameters"]:
                parameters[mapping["parameters"][k]] = v

        # instantiate the model and pass in the corrected parameters
        return getattr(self.pdk, mapping["name"])(**parameters)

    def parse(self, string: str) -> "Circuit":
        from simphony.models import Subcircuit
        from simphony.plugins.siepic import load_spi_from_string
        from simphony.simulators import SweepSimulator

        data = load_spi_from_string(string)

        # for now, circuits only ever contain one subcircuit
        # if that ever changes, we will need to update this
        for sub in data["subcircuits"]:
            # instantiate components in the subcircuit
            components = []
            for c in sub["components"]:
                component = self._instantiate_component(c)
                component.rename_pins(*c["ports"])
                components.append(component)

            # connect the netlists
            # compare indices so we only connect components once
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components):
                    if comp1 != comp2 and i <= j:
                        comp1.interface(comp2)

            # create a subcircuit instance
            # not permanent because we end up returning the wrapped circuit
            subcircuit = Subcircuit(components[0].circuit, sub["name"], permanent=False)
            subcircuit.rename_pins(*sub["ports"])

        # setup a SweepSimulator if one is active
        for analysis in data["analyses"]:
            if analysis["definition"]["input_parameter"] == "start_and_stop":
                _, input = analysis["params"]["input"][0].split(",")
                _, output = analysis["params"]["output"].split(",")
                start = analysis["params"]["start"]
                stop = analysis["params"]["stop"]
                points = int(analysis["params"]["number_of_points"])
                mode = (
                    "wl"
                    if analysis["definition"]["input_unit"] == "wavelength"
                    else "freq"
                )

                simulator = SweepSimulator(start, stop, points)
                simulator.mode = mode
                simulator.multiconnect(subcircuit[output], subcircuit[input])

        # no reason to include the subcircuit component for now
        return subcircuit._wrapped_circuit
