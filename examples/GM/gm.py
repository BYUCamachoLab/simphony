import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
# import simphony.errors as errors
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.ann as lib
import simphony.simulation as sim

inputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]
wg1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
dc1 = [inst(dev.ebeam_bdc_te1550) for _ in range(2)]
wg_inner1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
# crossover = inst(lib.crossover)
wg_inner2 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
wg_outer = [inst(dev.ebeam_wg_integral_1550, extras={'length':300e-6}) for _ in range(2)]
dc2 = [inst(dev.ebeam_bdc_te1550) for _ in range(2)]
wg3 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
outputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]

connections = []
for i in range(4):
    connections.append([inputs[i], 1, wg1[i], 0])

