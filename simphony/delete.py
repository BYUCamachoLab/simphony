from matplotlib import pyplot as plt
import numpy as np
# from utils import add_settings_to_netlist, get_settings_from_netlist, netlist_to_graph, graph_to_netlist
import simphony.libraries.siepic as siepic
import simphony.libraries.analytic as analytic
from copy import deepcopy
from simphony.circuit import Circuit
from simphony.simulation import SParameterSimulation
import sax
# netlist={
#     "instances": {
#         "splitter": {
#             "component":"ybranch",
#             "settings":{
#                 "test_setting": 100,
#             },
#         },
#         "combiner": "ybranch",  
#         "top1": "waveguide",
#         "top2": "waveguide",
#         "bot1": "waveguide",
#         "bot2": "waveguide",

#         "pm1": "phase_modulator",
#         "pm2": "phase_modulator",

#         "vs1": "voltage_source",
#         "vs2": "voltage_source",
#         "vs3": "voltage_source",

#         "opamp":"opamp",

#         "vf1":"voltage_follower",
#         "vf2":"voltage_follower",
#         "vf3":"voltage_follower",
#     },
#     "connections": {
#         "splitter,port_2":"top1,o0",
#         "splitter,port_3":"bot1,o0",
#         "top2,o1":"combiner,port_2",   
#         "bot2,o1": "combiner,port_3",
#         "top1,o1":"pm1,o0",
#         "pm1,o1":"top2,o0",
#         "bot1,o1":"pm2,o0",
#         "pm2,o1":"bot2,o0",

#         "vs1,e0":"""vf3,e0;
#                       vf1,e0;""",

#         "vs3,e0":"""vf2,e0;
#                     opamp,inv""",
        
#         "vs2,e0":"opamp,ninv",
        
#         "vf2,e1":"opamp,vp",

#         "vf3,e1":"pm2,e0",
#         "vf1,e0":"opamp,vn",     

#         "opamp,vout":"pm1,e0",
#     },
#     "ports": {
#         "in": "splitter,port_1",
#         "out": "combiner,port_1",
#     }
# }

# models={
#     "ybranch": siepic.y_branch,
#     # "ybranch": analytic.optical_s_parameter(siepic.y_branch),
#     "waveguide": analytic.Waveguide,
#     "phase_modulator": analytic.OpticalModulator,
#     "voltage_source": analytic.VoltageSource,
#     "prng": analytic.PRNG,
#     "voltage_follower": analytic.VoltageFollower,
#     "opamp": analytic.OpAmp,
# }

# settings={ 
#     # "splitter": {"bad_setting": 10},
#     "top1": {"length": 5},
#     "top2": {"length": 5},
#     "bot1": {"length": 10},
# }

# ckt = Circuit(netlist, models, default_settings=settings)

netlist = {
    "instances": {
        "hr1": "half_ring",
        "hr2": "half_ring",
        "w1":  {"component":"waveguide",
                "settings":{
                    "length": 20,
                },
        },
        "w2": "waveguide",
    },
    "connections": {
        # "hr1,port_1": "hr2,port_1",
        # "hr1,port_3": "hr2,port_3",
        "hr1,port_1": "w1,o0",
        "w1,o1": "hr2,port_1",
        "hr1,port_3": "w2,o0",
        "w2,o1": "hr2,port_3",

    },
    "ports": {
        "o0": "hr1,port_2",
        "o1": "hr2,port_2",
        "o2": "hr1,port_4",
        "o3": "hr2,port_4",
    }
}

models = {
    "half_ring": siepic.half_ring,
    "waveguide": siepic.waveguide,
}

ckt = Circuit(netlist, models)
wl = np.linspace(1.5, 1.6, 1000)*1e-6
settings = {

}
sps = SParameterSimulation(ckt)
results = sps.run(wl,settings)
print("S-Parameters:")
print(results.s_parameters)
plt.plot(wl, np.abs(results.s_parameters[("o0","o2")])**2, label="S11")
plt.show()
plt.plot(wl, np.angle(results.s_parameters[("o0","o2")]), label="S11")
plt.show()


# netlist = {
#     "instances": {
#         "hr1": "half_ring",
#         "hr2": "half_ring",
#     },
#     "connections": {
#         "hr1,port_1": "hr2,port_1",
#         "hr1,port_3": "hr2,port_3",

#     },
#     "ports": {
#         "o0": "hr1,port_2",
#         "o1": "hr2,port_2",
#         "o2": "hr1,port_4",
#         "o3": "hr2,port_4",
#     }
# }

# models = {
#     "half_ring": siepic.half_ring,
# }

# circuit,_ = sax.circuit(
#     netlist=netlist,
#     models=models,
# )
# wl = np.linspace(1.5, 1.6, 200)
# model_settings = {
#     "wl": wl,
# }
# s_params = circuit(**model_settings)
# plt.plot(wl, np.angle(s_params[("o0","o1")]), label="S11")
# plt.show()
