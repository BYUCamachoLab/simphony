'''
Test functions for simphony.core
'''

import pytest
import copy
import numpy as np
import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.devices as dev
import simphony.simulation as sim

class TestClass:
    @classmethod
    def setup(cls):
        core.clear_models()

    def test_ComponentModel_duplicity(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        with pytest.raises(errors.DuplicateModelError):
            rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
            rr2 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
        with pytest.raises(errors.DuplicateModelError):
            rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
            rr2 = copy.deepcopy(rr1)

    def test_ComponentModel_cachable(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        rr1 = core.ComponentModel("ring_resonator_1", 4, fake_s_params, cachable=True)
        rr2 = core.ComponentModel("ring_resonator_2", 4, fake_s_params, cachable=True)
        assert fake_s_params == rr1.get_s_parameters()
        assert fake_s_params == rr1.get_s_parameters(fake_keyword=3, faker_keyword="long")
        with pytest.raises(ValueError):
            rr3 = core.ComponentModel("ring_resonator_3", 4, cachable=True)

    def test_ComponentModel_uncachable(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        rr1 = core.ComponentModel("ring_resonator_1", 4, fake_s_params, cachable=False)
        with pytest.raises(NotImplementedError):
            rr1.get_s_parameters()
        rr2 = core.ComponentModel("ring_resonator_2", 4, fake_s_params, cachable=False)
        rr3 = core.ComponentModel("ring_resonator_3", 4, cachable=False)

    def test_ComponentInstance_cachableModel(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
        ci1 = core.ComponentInstance(rr1, [0,1,2,3], {'extras':'should be ignored'})
        assert rr1.get_s_parameters() == fake_s_params
        assert rr1.get_s_parameters() == ci1.get_s_parameters()
        assert ci1.get_s_parameters() == fake_s_params

    def test_ComponentInstance_uncachableModel(self):
        wg1 = core.ComponentModel("variable_length_waveguide", 2, cachable=False)
        def wg_s_parameters(freq, length, height):
            # Some random equation just to allow us to see the effects of the function
            return height*np.sin(freq)+length
        wg1.get_s_parameters = wg_s_parameters

        extras = {
            'freq': np.linspace(0,2*np.pi),
            'length': 2,
            'height':0.5
        }
        ci1 = core.ComponentInstance(wg1, [0,1], extras)
        expected = wg_s_parameters(extras['freq'], extras['length'], extras['height'])
        assert np.array_equal(expected, ci1.get_s_parameters())
        assert np.array_equal(expected, wg1.get_s_parameters(**extras))

    # def test_core(self):
    #     rr = core.ComponentModel("ring_resonator", ([1500, 1550, 1600], [0,0,0]), cachable=True)
    #     nl = core.Netlist()
    #     c1 = core.ComponentInstance(rr, [0,1,2,3], lay_x=3.1, lay_y=4)
    #     c2 = core.ComponentInstance(rr, [4,1,2,5])
    #     nl.add_component(c1)
    #     nl.add_component(c2)
    #     assert len(nl.components) == 2

class TestCircuit:

    @classmethod
    def setup(cls):
        core.clear_models()

    def test_4Port_Circuit(self):
        # Device is modeled after fabRun1/A4
        y = dev.ebeam_y_1550()
        gc = dev.ebeam_gc_te1550()
        wg = dev.ebeam_wg_integral_1550()
        bdc = dev.ebeam_bdc_te1550()
        term = dev.ebeam_terminator_te1550()

        gc1 = core.ComponentInstance(gc)
        gc2 = core.ComponentInstance(gc)
        gc3 = core.ComponentInstance(gc)
        gc4 = core.ComponentInstance(gc)

        y1 = core.ComponentInstance(y)
        y2 = core.ComponentInstance(y)
        y3 = core.ComponentInstance(y)

        bdc1 = core.ComponentInstance(bdc)
        bdc2 = core.ComponentInstance(bdc)

        term1 = core.ComponentInstance(term)

        wg1 = core.ComponentInstance(wg, extras={'length':165.51e-6})
        wg2 = core.ComponentInstance(wg, extras={'length':247.73e-6})
        wg3 = core.ComponentInstance(wg, extras={'length':642.91e-6})
        wg4 = core.ComponentInstance(wg, extras={'length':391.06e-6})

        wg5 = core.ComponentInstance(wg, extras={'length':10.45e-6})
        wg6 = core.ComponentInstance(wg, extras={'length':10.45e-6})
        wg7 = core.ComponentInstance(wg, extras={'length':10.45e-6})
        wg8 = core.ComponentInstance(wg, extras={'length':10.45e-6})

        wg9 = core.ComponentInstance(wg, extras={'length':162.29e-6})
        wg10 = core.ComponentInstance(wg, extras={'length':205.47e-6})

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
        y = dev.ebeam_y_1550()
        gc = dev.ebeam_gc_te1550()
        wg = dev.ebeam_wg_integral_1550()
        bdc = dev.ebeam_bdc_te1550()
        term = dev.ebeam_terminator_te1550()

        y1 = core.ComponentInstance(y)
        y2 = core.ComponentInstance(y)
        wg1 = core.ComponentInstance(wg, extras={'length':50e-6})
        wg2 = core.ComponentInstance(wg, extras={'length':150e-6})

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
