# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import simphony
# from simphony.elements import Element
# from simphony.netlist import Subcircuit

# class RingResonator(Element):
#     nodes = ('n1', 'n2', 'n3', 'n4')

#     def __init__(self, radius):
#         """
#         It is up to the library creator to determine the units they accept
#         as parameters, or whether they accept other parameters/flags to 
#         indicate the units of other parameters.

#         By default, Simphony assumes basic SI units throughout (i.e. meters,
#         Hertz, etc.).
#         """
#         self.radius = radius

#     def s_params(self):
#         # Note that loading .npz files allows for an approximately 
#         # 3.5x speedup over parsing text files.
#         # Do some calculation here that returns the s_parameters over
#         # some frequency range.
#         pass


# circuit = Subcircuit('Add-Drop Filter')
# circuit.add([
#     ("ring 10um", RingResonator(radius=10e-6)),
#     (None, RingResonator(radius=11e-6)),
#     ("ring 12um", RingResonator(radius=12e-6)),
# ])

# r10 = circuit["ring 10um"]

# circuit.connect("ring 10um", 'n3', "ring 12um", 'n4')
# circuit.connect([])
# # circuit.label()


# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np

# import simphony
# import simphony.library.ebeam as ebeam
# from simphony.netlist import Subcircuit
# from simphony.simulation import SweepSimulation

# def MZIFactory(name, dl):
#     length = 50e-6

#     circuit = Subcircuit(name)
#     circuit.add([
#         ebeam.ebeam_y_1550('splitter'),
#         ebeam.ebeam_y_1550('recombiner'),
#         ebeam.ebeam_wg_integral_1550('wg_long', length=length+dl), # can optionally include ne=10.1, ng=1.3
#         ebeam.ebeam_wg_integral_1550('wg_short', length=length),
#     ])

#     circuit['splitter'].rename_nodes(('input', 'out1', 'out2'))
#     circuit['recombiner'].rename_nodes(('output', 'in2', 'in1'))

#     circuit.connect_many([
#         ('splitter', 'out1', 'wg_long', 'n1'),
#         ('splitter', 'out2', 'wg_short', 'n1'),
#         ('recombiner', 'in1', 'wg_long', 'n2'),
#         ('recombiner', 'in2', 'wg_short', 'n2'),
#     ])

#     # circuit.rename_nodes('n1', 'n2',)
#     return circuit

# circuit = Subcircuit('MZI Cascade')
# circuit.add([
#     MZIFactory('input', 50e-6),
#     MZIFactory('middle', 100e-6),
#     MZIFactory('end', 150e-6),
# ])

# # Run a simulation on the netlist.
# simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
# simulation.simulate()





# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import numpy as np

# import simphony
# import simphony.library.ebeam as ebeam
# from simphony.netlist import Subcircuit
# from simphony.simulation import SweepSimulation

# circuit = Subcircuit('MZI')
# circuit.add([
#     ebeam.ebeam_y_1550('splitter'),
#     ebeam.ebeam_y_1550('recombiner'),
#     ebeam.ebeam_wg_integral_1550('wg_long', length=150e-6), # can optionally include ne=10.1, ng=1.3
#     ebeam.ebeam_wg_integral_1550('wg_short', length=50e-6),
# ])

# circuit['splitter'].rename_nodes(('in1', 'out1', 'out2'))
# circuit['recombiner'].rename_nodes(('out1', 'in2', 'in1'))

# circuit.connect_many([
#     ('splitter', 'out1', 'wg_long', 'n1'),
#     ('splitter', 'out2', 'wg_short', 'n1'),
#     ('recombiner', 'in1', 'wg_long', 'n2'),
#     ('recombiner', 'in2', 'wg_short', 'n2'),
# ])

# # Run a simulation on the netlist.
# simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
# simulation.simulate()


from simphony.elements import Element, PinList
a = Element()
a.nodes = PinList('a', 'b', 'c', 'd')
print(a.nodes)






import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import simphony
import simphony.library.ebeam as ebeam
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation

circuit = Subcircuit('MZI')
circuit.add([
    ebeam.ebeam_y_1550('splitter'),
    ebeam.ebeam_y_1550('recombiner'),
    ebeam.ebeam_wg_integral_1550('wg_long', length=150e-6), # can optionally include ne=10.1, ng=1.3
    ebeam.ebeam_wg_integral_1550('wg_short', length=50e-6),
])

circuit['splitter'].pins = PinList('in1', 'out1', 'out2')
circuit['recombiner'].pins = PinList('out1', 'in2', 'in1')

circuit.connect_many([
    ('splitter', 'out1', 'wg_long', 'n1'),
    ('splitter', 'out2', 'wg_short', 'n1'),
    ('recombiner', 'in1', 'wg_long', 'n2'),
    ('recombiner', 'in2', 'wg_short', 'n2'),
])

# # Run a simulation on the netlist.
# simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
# simulation.simulate()