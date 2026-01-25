"""
_netlist module provides functionality for netlists that do not contain sax models.
Really this module is only here to allow simphony.libraries.ideal.s_parameter access to this functionality,
while still allowing simphony.libraries.ideal.s_parameter to be used in the public simphony.circuit.netlist module
"""
from sax import AnyNetlist, InstanceName, Ports, Connections, Models
import sax
from typing import TypeAlias, TypedDict
from simphony.component.component import Component
from typing_extensions import NotRequired
import networkx as nx
import jax.numpy as jnp
import yaml
from copy import deepcopy
from simphony.component.pcell import PCell
from simphony.simulation.simulation import SimulationMode

# TODO: modify so that the Component field may be a dict (for specifying the groups)
ElaboratedInstances: TypeAlias = dict[InstanceName, Component]
"""A mapping from instance names to their instantiated component model."""

InstantiatedFlatNetlist = TypedDict(
        "Netlist",
        {
            "instances": ElaboratedInstances,
            "connections": NotRequired[Connections], # TODO: Add simphony type for connections, since ours are more general
            "ports": Ports,
            # "nets": NotRequired[Nets],
            # "placements": NotRequired[Placements],
            # "settings": NotRequired[Settings],
        },
    )

def _instantiate_netlist(
    netlist,
    models,
    settings,
    simulation_mode: SimulationMode,
    directed: bool,
    default_modes: tuple,
)->InstantiatedFlatNetlist:
    """
    Generated instantiated netlist from netlist which does not contain sax models
    """
    instantiated_recursive_netlist = sax.netlist(deepcopy(netlist))
    for _, subnetlist in instantiated_recursive_netlist.items():
        ## TODO: Actually add the settings to the netlist if desired
        _add_settings_to_netlist(subnetlist) # Just to normalize, we will use the settings the user provided later
    instantiated_recursive_netlist = sax.netlist(instantiated_recursive_netlist)
    instantiated_flat_netlist = sax.flatten_netlist(instantiated_recursive_netlist)

    ## Make unique, instantiated models for each instance. Put the instantiated models in the netlist metadata
    for instance_name, instance_data in instantiated_flat_netlist["instances"].items():
        print(instance_name)
        component_name = instance_data["component"]
        instance_settings = settings.get(instance_name, {})
        uninstantiated_model = models[component_name]

        if issubclass(uninstantiated_model, PCell):
            instance_data["model"] = uninstantiated_model(simulation_mode, **instance_settings)
        elif issubclass(uninstantiated_model, Component):
            instance_data["model"] = uninstantiated_model(simulation_mode, **instance_settings)
    
    ## Get instantiated flat nelist from any pcells and splice them into instatiated_netlist issubclass(models['mzi'], PCell)
    ## When splicing in, make sure that there are no conflicts with model names
    instantiated_flat_netlist_no_pcells = deepcopy(instantiated_flat_netlist)

    for instance_name, instance_data in instantiated_flat_netlist['instances'].items():
        instantiated_model = instance_data['model']
        if isinstance(instantiated_model, PCell):
            ### TODO: Give _instantiated_netlist the proper arguments
            instantiated_model._instantiated_netlist(simulation_mode, directed=directed, default_modes=default_modes)
            ### TODO: Stitch the netlist
            pcell_netlist = ...


    ## Return a new, flat instantiated netlist with no pcells
    return {}

def _add_settings_to_netlist(netlist, settings=None):
    if settings is None:
        settings = {}
    # Ensure Instance Name corresponds to a dictionary with the proper format
    for instance_name, model in netlist["instances"].items():
        if isinstance(model, str):
            netlist["instances"][instance_name] = {"component": model, "settings": {}}
        elif isinstance(model, dict):
            if not netlist["instances"][instance_name].get("settings"):
                netlist["instances"][instance_name]["settings"] = {}

    for instance_name, instance_settings in settings.items():
        netlist["instances"][instance_name]["settings"].update(instance_settings)
