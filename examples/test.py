from simphony.circuit.circuit import Circuit
from simphony.libraries.ideal.photonic_circuits import mzi_lattice_filter
from simphony.libraries.siepic import grating_coupler
# from simphony.simulation.simulation import SimulationMode
from simphony.simulation.block_mode import BlockModeSimulationParameters
# from simphony.simulation.sample_mode import SampleModeSimulationParameters
# from simphony.simulation.s_parameter import SParameterSimulationParameters
from simphony.libraries.ideal.sources import CWLaser

netlist = {
    "instances": {
        "lf1": "lattice_filter",
        "lf2": "lattice_filter",
        "gc1": "grating_coupler",
        "gc2": "grating_coupler",
        "laser": "cw_laser",
    },
    "connections": {
        "laser,o0": "gc1,o0",
        "gc1,o1": "lf1,o0",
        "lf1,o1": "lf2,o0",
        "lf2,o1": "gc2,o1",
    },
    "ports": {
        # "in": "gc1,o0",
        "out": "gc2,o0",
    }
}

models = {
    "lattice_filter": mzi_lattice_filter(3),
    "grating_coupler": grating_coupler,
    "cw_laser": CWLaser,
}

s_parameter_settings = {
    "lf1": {

    },
    "lf2": {

    },
    "gc1": {
        "pol": "te",
        "thickness": 230.0,
        "dwidth": -20,
    },
    "gc2": {
        "pol": "tm",
        "thickness": 210.0,
        "dwidth": 20,
    },
}


sample_mode_settings = {
    "lf1": {

    },
    "lf2": {

    },
    "gc1": {
        "pol": "te",
        "thickness": 230.0,
        "dwidth": -20,
    },
    "gc2": {
        "pol": "tm",
        "thickness": 210.0,
        "dwidth": 20,
    },
}

block_mode_settings = {
    "lf1": {

    },
    "lf2": {

    },
    "gc1": {
        "sax_settings": {
            "pol": "te",
            "thickness": 230.0,
            "dwidth": -20,
        },
    },
    "gc2": {
        "sax_settings": {
            "pol": "tm",
            "thickness": 210.0,
            "dwidth": 20,
        }
    },
}

circuit = Circuit(netlist, models)
# circuit.display(inline=True)
instantiated_circuit = circuit.instantiate(block_mode_settings, BlockModeSimulationParameters(), directed=True, fuse_models=True)
# instantiated_circuit = circuit.instantiate(s_parameter_settings, SParameterSimulationParameters())


instantiated_circuit.display(inline=False)