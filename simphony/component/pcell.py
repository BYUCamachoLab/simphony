import inspect
from copy import deepcopy

import sax

from simphony.component.component import Component
from simphony.simulation.simulation import SimulationParameters


### TODO: Decide whether PCells are Components or not, gotta love OOP
class PCell(Component):
    """Base class for parameterized circuit cells.

    A PCell is useful when a component's internal netlist depends on
    constructor settings or the active simulation mode. Circuit diagrams
    can treat the PCell as one component, while simulation expands it
    into its internal `netlist`, `models`, and `settings`.

    Subclasses should define external `ports` at the class level. During
    `__init__`, the subclass should populate `self.netlist`,
    `self.models`, and `self.settings` with the internal circuit that
    should replace the PCell.

    The constructor is normally called by circuit instantiation rather
    than directly by users building a netlist.
    """

    ports = None

    netlist = None
    models = None
    settings = None

    def __repr__(self):
        return f"<{type(self).__name__} (PCell obj)>"

    def __init__(
        self,
        simulation_mode: SimulationParameters,
        **kwargs,
    ): ...

    def _instantiated_netlist(
        self,
        simulation_parameters: SimulationParameters,
    ):
        """Expand this PCell into an instantiated internal netlist.

        Raw SAX callables in `self.models` are wrapped as
        `SParameterElement` components, then the internal netlist is
        instantiated with the active simulation parameters.
        """
        if self.netlist is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `netlist` before instantiation"
            )

        if self.models is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `models` before instantiation"
            )

        from simphony.circuit.netlist import instantiate_netlist

        new_netlist, new_models, new_settings = _convert_sax_models(
            self.netlist,
            self.models,
            self.settings,
        )

        return instantiate_netlist(
            new_netlist,
            new_models,
            new_settings,
            simulation_parameters,
        )


def _convert_sax_models(netlist, models, settings):
    """Wrap raw SAX models inside a PCell as `SParameterElement` classes.

    PCells may define their internal `models` dictionary using plain SAX
    callables. They are wrapped with `optical_s_parameter` so that port
    metadata and settings are handled uniformly; port directionality is
    assigned later, once the whole circuit has been flattened.

    Returns
    -------
    tuple
        `(new_netlist, new_models, new_settings)` ready for recursive
        instantiation.
    """
    from simphony.circuit.netlist import add_settings_to_netlist
    from simphony.libraries.ideal.s_parameters import (
        optical_s_parameter,
        wrap_sax_settings,
    )

    netlist = sax.netlist(deepcopy(netlist))
    for _, subnetlist in netlist.items():
        add_settings_to_netlist(subnetlist)
    netlist = sax.flatten_netlist(sax.netlist(netlist))

    new_models = {
        model_name: model if inspect.isclass(model) else optical_s_parameter(model)
        for model_name, model in models.items()
    }
    new_settings = wrap_sax_settings(
        netlist["instances"].keys(),
        lambda instance: new_models[netlist["instances"][instance]["component"]],
        deepcopy(settings) if settings is not None else {},
    )

    return netlist, new_models, new_settings
