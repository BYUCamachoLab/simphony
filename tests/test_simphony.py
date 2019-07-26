import pytest
import copy
import numpy as np
import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.ebeam as dev
import simphony.simulation as sim

class TestNetlist:

    def test_4Port_Circuit(self):
        gc1 = core.ComponentInstance(dev.ebeam_gc_te1550)
        gc2 = core.ComponentInstance(dev.ebeam_gc_te1550)
        gc3 = core.ComponentInstance(dev.ebeam_gc_te1550)
        gc4 = core.ComponentInstance(dev.ebeam_gc_te1550)

        y1 = core.ComponentInstance(dev.ebeam_y_1550)
        y2 = core.ComponentInstance(dev.ebeam_y_1550)
        y3 = core.ComponentInstance(dev.ebeam_y_1550)

        bdc1 = core.ComponentInstance(dev.ebeam_bdc_te1550)
        bdc2 = core.ComponentInstance(dev.ebeam_bdc_te1550)

        term1 = core.ComponentInstance(dev.ebeam_terminator_te1550)

        wg1 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':165.51e-6})
        wg2 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':247.73e-6})
        wg3 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':642.91e-6})
        wg4 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':391.06e-6})

        wg5 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':10.45e-6})
        wg6 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':10.45e-6})
        wg7 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':10.45e-6})
        wg8 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':10.45e-6})

        wg9 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':162.29e-6})
        wg10 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':205.47e-6})

        connections = []
        connections.append([gc1, 0, wg1, 1])
        connections.append([gc3, 0, wg2, 1])
        connections.append([bdc1, 3, wg1, 0])
        connections.append([bdc1, 2, wg2, 0])
        connections.append([gc2, 0, y1, 0])
        connections.append([y1, 1, wg3, 0])
        connections.append([y1, 2, wg4, 0])
        connections.append([y2, 0, wg4, 1])
        connections.append([y3, 0, wg3, 1])
        connections.append([y2, 1, wg5, 1])
        connections.append([bdc1, 0, wg5, 0])
        connections.append([bdc1, 1, wg6, 1])
        connections.append([y3, 2, wg6, 0])
        connections.append([y2, 2, wg7, 0])
        connections.append([y3, 1, wg8, 1])
        connections.append([bdc2, 2, wg7, 1])
        connections.append([bdc2, 3, wg8, 0])
        connections.append([bdc2, 0, wg9, 0])
        connections.append([term1, 0, wg9, 1])
        connections.append([bdc2, 1, wg10, 0])
        connections.append([gc4, 0, wg10, 1])

        nl = core.Netlist()
        nl.load(connections, formatter='ll')

        for item in nl.components:
            print(item.model)

        simu = sim.Simulation(nl)
        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 2, 0])**2)
        plt.subplot(222)
        plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 2, 1])**2)
        plt.subplot(223)
        plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 2, 2])**2)
        plt.subplot(224)
        plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 2, 3])**2)
        plt.suptitle("A4")
        plt.show()

    def test_mzi(self):
        y1 = core.ComponentInstance(dev.ebeam_y_1550)
        y2 = core.ComponentInstance(dev.ebeam_y_1550)
        wg1 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':50e-6})
        wg2 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':150e-6})

        c1 = [y1, y1, y2, y2]
        p1 = [1, 2, 2, 1]
        c2 = [wg1, wg2, wg1, wg2]
        p2 = [0, 0, 1, 1]
        con = zip(c1, p1, c2, p2)

        nl = core.Netlist()
        nl.load(con, formatter='ll')
        simu = sim.Simulation(nl)

        import matplotlib.pyplot as plt
        plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 0, 1])**2)
        plt.title("MZI")
        plt.show()