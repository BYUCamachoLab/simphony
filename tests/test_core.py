'''
Test functions for simphony.core
'''

import pytest
import copy
import numpy as np
import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.devices as dev

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
        ci1 = core.ComponentInstance(rr1, [0,1,2,3], 0, 1, {'extras':'should be ignored'})
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
        ci1 = core.ComponentInstance(wg1, [0,1], 0, 1, extras)
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
    def test_4Port_Circuit(self):
        y = dev.ebeam_y_1550()
        gc = dev.ebeam_gc_te1550()
        wg = dev.ebeam_wg_integral_1550()
        bdc = dev.ebeam_bdc_te1550()
        term = dev.ebeam_terminator_te1550()

        # gc1 = 

        c1 = []
        p1 = []
        c2 = []
        p2 = []