from simphony.libraries.ideal.modulators import OpticalModulator
from simphony.libraries.ideal.sources import VoltageSource
from simphony.libraries.ideal.electrical_circuits import VoltageFollower
from simphony.libraries.ideal.s_parameters import optical_s_parameter_placeholder
from simphony.libraries.siepic import y_branch, waveguide

import sax
import numpy as np

instances = {
    "splitter": "y_branch",
    "combiner": "y_branch",
    "wg1": "waveguide",
    "wg2": "waveguide",
    "mod1": "modulator",
    "mod2": "modulator",
    "vs1": "voltage_source",
    "vf1": "voltage_follower",
    "vs2": "voltage_source",
}
connections = {
    "splitter,port_2": "wg1,o0",
    "wg1,o1": "mod1,o0",
    "mod1,o1": "combiner,port_2",
    "splitter,port_3": "wg2,o0",
    "wg2,o1": "mod2,o0",
    "mod2,o1": "combiner,port_3",
    "vs1,e0": "vf1,e0",
    "vf1,e1": "mod1,e0",
    "vs2,e0": "mod2,e0"
}
ports = {
    "in": "splitter,port_1",
    "out": "combiner,port_1",
}

netlist = {
    "instances": instances,
    "connections": connections,
    "ports": ports,
}

models = {
    "y_branch": y_branch,
    "waveguide": waveguide,
    "modulator": OpticalModulator,
    "voltage_source": VoltageSource,
    "voltage_follower": VoltageFollower,
}

ckt_settings = {
    "splitter": {},
    "combiner": {},
    "wg1": {"length":10.0},
    "wg2": {"length":30.0},
    "mod1": {
        "phase_coefficients": np.array([0.0, 0.0, np.pi/2, 0]),
        # "phase_coefficients": np.array([0.0, 0.0, 0.0, 0])
    },
    "mod2": {},
    "vs1": {
        "steady_state_voltage": 3.0,
    },
    "vs2": {
        "steady_state_voltage": 0.0,
    },
}

from simphony.circuit.circuit import Circuit
from simphony.simulation.s_parameter import SParameterSimulationParameters
# from simphony.simulation.block_mode import BlockModeSimulationParameters

simulation_parameters = SParameterSimulationParameters()
# simulation_parameters = BlockModeSimulationParameters()
mzi_circuit = Circuit(netlist, models)
mzi_circuit.display()
mzi_circuit.instantiate(ckt_settings, simulation_parameters)

from simphony.simulation.s_parameter import SParameterSimulation
import numpy as np
import matplotlib.pyplot as plt

# wl = np.linspace(1.5e-6, 1.6e-6, 1000)
wl = np.linspace(1.5, 1.6, 1000)*1e-6
sparam_simulation = SParameterSimulation(mzi_circuit, ckt_settings, simulation_parameters)
sim_result = sparam_simulation.run(wl=wl)

s_params_1d = sim_result.s_parameters[('in', 'out')]

# from simphony.circuit.circuit import Circuit
# from simphony.libraries.ideal.photonic_circuits import mzi_lattice_filter
# from simphony.libraries.siepic import grating_coupler
# # from simphony.simulation.simulation import SimulationMode
# from simphony.simulation.block_mode import BlockModeSimulationParameters
# from simphony.simulation.sample_mode import SampleModeSimulationParameters
# # from simphony.simulation.s_parameter import SParameterSimulationParameters
# from simphony.libraries.ideal.sources import CWLaser

# netlist = {
#     "instances": {
#         "lf1": "lattice_filter",
#         "lf2": "lattice_filter",
#         "gc1": "grating_coupler",
#         "gc2": "grating_coupler",
#         "laser": "cw_laser",
#     },
#     "connections": {
#         "laser,o0": "gc1,o0",
#         "gc1,o1": "lf1,o0",
#         "lf1,o1": "lf2,o0",
#         "lf2,o1": "gc2,o1",
#     },
#     "ports": {
#         # "in": "gc1,o0",
#         "gc_out": "gc2,o0",
#     }
# }

# models = {
#     "lattice_filter": mzi_lattice_filter(2),
#     "grating_coupler": grating_coupler,
#     "cw_laser": CWLaser,
# }

# s_parameter_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "pol": "te",
#         "thickness": 230.0,
#         "dwidth": -20,
#     },
#     "gc2": {
#         "pol": "tm",
#         "thickness": 210.0,
#         "dwidth": 20,
#     },
# }


# sample_mode_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "pol": "te",
#         "thickness": 230.0,
#         "dwidth": -20,
#     },
#     "gc2": {
#         "pol": "tm",
#         "thickness": 210.0,
#         "dwidth": 20,
#     },
# }

# block_mode_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "sax_settings": {
#             "pol": "te",
#             "thickness": 230.0,
#             "dwidth": -20,
#         },
#     },
#     "gc2": {
#         "sax_settings": {
#             "pol": "tm",
#             "thickness": 210.0,
#             "dwidth": 20,
#         }
#     },
# }

# tracked_ports = {
#     # "gc1": "gc1,o1",
#     # "lf1_in": "lf1,o0",
#     "lf1_out": "lf1,o1",
#     "lf2": "lf2,o1",
# }

# circuit = Circuit(netlist, models)
# instantiated_circuit = circuit.instantiate(sample_mode_settings, SampleModeSimulationParameters(), tracked_ports=tracked_ports, directed=False)


# instantiated_circuit.display()
# from simphony.circuit.circuit import Circuit
# from simphony.libraries.ideal.photonic_circuits import mzi_lattice_filter
# from simphony.libraries.siepic import grating_coupler
# # from simphony.simulation.simulation import SimulationMode
# from simphony.simulation.block_mode import BlockModeSimulationParameters
# from simphony.simulation.sample_mode import SampleModeSimulationParameters
# # from simphony.simulation.s_parameter import SParameterSimulationParameters
# from simphony.libraries.ideal.sources import CWLaser

# netlist = {
#     "instances": {
#         "lf1": "lattice_filter",
#         "lf2": "lattice_filter",
#         "gc1": "grating_coupler",
#         "gc2": "grating_coupler",
#         "laser": "cw_laser",
#     },
#     "connections": {
#         "laser,o0": "gc1,o0",
#         "gc1,o1": "lf1,o0",
#         "lf1,o1": "lf2,o0",
#         "lf2,o1": "gc2,o1",
#     },
#     "ports": {
#         # "in": "gc1,o0",
#         "gc_out": "gc2,o0",
#     }
# }

# models = {
#     "lattice_filter": mzi_lattice_filter(1),
#     "grating_coupler": grating_coupler,
#     "cw_laser": CWLaser,
# }

# s_parameter_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "pol": "te",
#         "thickness": 230.0,
#         "dwidth": -20,
#     },
#     "gc2": {
#         "pol": "tm",
#         "thickness": 210.0,
#         "dwidth": 20,
#     },
# }


# sample_mode_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "pol": "te",
#         "thickness": 230.0,
#         "dwidth": -20,
#     },
#     "gc2": {
#         "pol": "tm",
#         "thickness": 210.0,
#         "dwidth": 20,
#     },
# }

# block_mode_settings = {
#     "lf1": {

#     },
#     "lf2": {

#     },
#     "gc1": {
#         "sax_settings": {
#             "pol": "te",
#             "thickness": 230.0,
#             "dwidth": -20,
#         },
#     },
#     "gc2": {
#         "sax_settings": {
#             "pol": "tm",
#             "thickness": 210.0,
#             "dwidth": 20,
#         }
#     },
# }

# tracked_ports = {
#     "gc1": "gc1,o1",
#     "lf1_in": "lf1,o0",
#     "lf1_out": "lf1,o1",
#     "lf2": "lf2,o1",
# }

# circuit = Circuit(netlist, models)
# instantiated_circuit = circuit.instantiate(block_mode_settings, BlockModeSimulationParameters(), tracked_ports=tracked_ports, directed=True)


# instantiated_circuit.display()

# from simphony.simulation.block_mode import BlockModeSimulation, BlockModeSimulationParameters
# tracked_ports = {
#     "gc1": "gc1,o1",
#     "lf1": "lf1,o1",
#     "lf2": "lf2,o1",
# }
# circuit = Circuit(netlist, models)
# block_mode_simulation_parameters = BlockModeSimulationParameters(mode_identifiers=["TE", "TM"], use_speed_up=True)
# block_mode_simulation = BlockModeSimulation(circuit, block_mode_settings, tracked_ports, block_mode_simulation_parameters)
# block_mode_simulation_result = block_mode_simulation.run()