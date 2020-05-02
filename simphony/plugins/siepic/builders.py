# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import importlib

from simphony.elements import Model
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation
from simphony.plugins.siepic.mapping import rearg
from simphony.tools import get_subclasses

def get_components(libraries):
    """
    Parameters
    ----------
    libraries : str

    Returns
    -------
    components : dict
    """
    components = {}
    for lib in libraries:
        try:
            model_lib = importlib.import_module(lib)
        except ImportError:
            raise
        models = list(get_subclasses(Model))
        for model in models:
            components[model.__name__] = model
    return components

def build_circuit(data, libraries):
    """
    Parameters
    ----------
    data : dict
        The dictionary defining all the circuits and analyzers, as exported
        by SiEPIC and parsed by the parser.
    libraries : list or str
        A string or list of strings of python module names containing the 
        model libraries for the components in the circuit.

    Returns
    -------
    built : dict
        A dictionary of constructed Python objects, with the following keys:
            - `circuits`: dictionary of circuit names to their corresponding
              instantiated Subcircuit objects.
            - `subcircuits`: instantiated Subcircuit objects for all 
              subcircuits found in the spice data.
            - `analyses`: instantiated Simulation objects for all network
              analyzers found in the spice data.
    """
    # Make sure `libraries` is a list
    libraries = libraries if type(libraries) is list else [libraries]

    # Create a dictionary from component names to their models
    available_comps = get_components(libraries)

    # Create all the defined subcircuits
    # [`name`, `ports`, `components`, `params`]
    subcircuits = {}
    for sub in data['subcircuits']:
        circuit = Subcircuit(sub['name'])
        connections = {}

        # Create all the components in the subcircuit and compile all the connections
        # [`name`, `model`, `ports`, `params`]
        for component in sub['components']:
            # Add each component
            kwargs = rearg(component['model'], component['params'])
            circuit.add([(available_comps[component['model']](**kwargs), component['name'])])
            circuit.elements[component['name']].pins = tuple(pin for pin in component['ports'])

            # Track all it's nets
            for port in component['ports']:
                if port in connections:
                    connections[port].append(component['name'])
                else:
                    connections[port] = [component['name']]
        
        # Create all the connections between components
        for port, comps in connections.items():
            circuit.connect(comps[0], port, comps[1], port) if len(comps) == 2 else None

        subcircuits[sub['name']] = circuit

    # Create circuits, since they're composed of subcircuits
    # [`name`, `ports`, `subcircuits`, `params`]
    circuits = {}
    # TODO: What if one day a circuit contains more than one subcircuit?
    for circ in data['circuits']:
        subs = subcircuits[circ['subcircuits']]
        if len(circ['ports']) != len(subs.pins):
            raise ValueError('Ports on circuit do not match ports on subcircuit.')
        circuits[circ['name']] = subs

    # Create all the simulation objects
    # [`definition`, `params`].
    analyses = []
    for analysis in data['analyses']:
        # [`input_unit`, `input_parameter`]
        if analysis['definition']['input_parameter'] == 'start_and_stop':
            # [`minimum_loss`, `analysis_type`, `multithreading`,
            #  `number_of_threads`, `orthogonal_identifier`, `start`, `stop`, 
            #  `number_of_points`, `input`, `output`]
            circ, inp = analysis['params']['output'].split(',')
            start = analysis['params']['start']
            stop = analysis['params']['stop']
            points = int(analysis['params']['number_of_points'])
            mode = 'wl' if analysis['definition']['input_unit'] == 'wavelength' else 'freq'
            sim = SweepSimulation(circuits[circ], start, stop, points, mode)
            analyses.append(sim)
    
    return {'circuits': circuits, 'subcircuits': subcircuits, 'analyses': analyses}
