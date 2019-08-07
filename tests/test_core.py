'''
Test functions for simphony.core
'''

import pytest
import copy
import numpy as np

import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.ebeam as dev
import simphony.simulation as sim
from simphony.core import register_component_model, deregister_component_model

class TestBase:
    def test_ComponentModel_duplicity(self):
        """ Tests whether you can have two models with the same name.

        Should raise an error since the ebeam library was already 
        imported, and this class shares the name of one of the existing
        devices.
        """
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        with pytest.raises(errors.DuplicateModelError):
            @register_component_model
            class ebeam_bdc_te1550(core.ComponentModel):
                ports = 4
                s_parameters = fake_s_params
                cachable = True

    def test_ComponentModel_cachable(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])
        
        @register_component_model
        class RingResonator(core.ComponentModel):
            ports = 4
            s_parameters = fake_s_params
            cachable = True
        
        assert fake_s_params == RingResonator.get_s_parameters()
        assert fake_s_params == RingResonator.get_s_parameters(fake_keyword=3, faker_keyword="long")

        deregister_component_model('RingResonator')

        with pytest.raises(errors.CachableParametersError):
            @register_component_model
            class RingResonator(core.ComponentModel):
                ports = 4
                cachable = True

    def test_ComponentModel_uncachable(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])

        with pytest.raises(errors.UncachableParametersError):
            @register_component_model
            class RingResonator(core.ComponentModel):
                ports = 4
                s_parameters = fake_s_params
                cachable = False

        with pytest.raises(errors.UncachableParametersError):
            @register_component_model
            class RingResonator(core.ComponentModel):
                ports = 4
                cachable = False

    def test_ComponentInstance_cachableModel(self):
        fake_s_params = ([1500, 1550, 1600], [0,0,0])

        @register_component_model
        class RingResonator(core.ComponentModel):
            ports = 4
            s_parameters = fake_s_params
            cachable = True

        ci1 = core.ComponentInstance(RingResonator, [0,1,2,3], {'extras':'should be ignored'})
        assert RingResonator.get_s_parameters() == fake_s_params
        assert RingResonator.get_s_parameters() == ci1.get_s_parameters()
        assert ci1.get_s_parameters() == fake_s_params
        deregister_component_model('RingResonator')

    def test_ComponentInstance_uncachableModel(self):
        @register_component_model
        class Waveguide(core.ComponentModel):
            ports = 2
            cachable = False

            @classmethod
            def s_parameters(cls, freq, length, height):
                # Some random equation just to allow us to see the effects of the function
                return height*np.sin(freq)+length

        extras = {
            'freq': np.linspace(0,2*np.pi),
            'length': 2,
            'height':0.5
        }
        ci1 = core.ComponentInstance(Waveguide, [0,1], extras)
        expected = Waveguide.s_parameters(extras['freq'], extras['length'], extras['height'])
        assert np.array_equal(expected, ci1.get_s_parameters())
        assert np.array_equal(expected, Waveguide.get_s_parameters(**extras))
        deregister_component_model('Waveguide')

class TestNetlist:
    @classmethod
    def setup(cls):
        bdc1 = core.ComponentInstance(dev.ebeam_bdc_te1550, [0, 1, 2, 3])
        term1 = core.ComponentInstance(dev.ebeam_terminator_te1550, [2])
        y1 = core.ComponentInstance(dev.ebeam_y_1550, [-1, 0, 1])
        dc1 = core.ComponentInstance(dev.ebeam_dc_halfring_te1550, [3, -2])
        cls.components = [bdc1, term1, y1, dc1]

    def test_netlist_InstancesFromComponentModels(self):
        @register_component_model
        class RingResonator(core.ComponentModel):
            ports = 4
            s_parameters = ([1500, 1550, 1600], [0,0,0])
            cachable = True
            
        nl = core.Netlist()
        c1 = core.ComponentInstance(RingResonator, [0,1,2,3], {'lay_x':3.1, 'lay_y':4})
        c2 = core.ComponentInstance(RingResonator, [4,1,2,5])
        nl.add_component(c1)
        nl.add_component(c2)
        assert len(nl.components) == 2
        deregister_component_model('RingResonator')

    def test_parsing_listoflists(self):
        bdc1 = core.ComponentInstance(dev.ebeam_bdc_te1550)
        term1 = core.ComponentInstance(dev.ebeam_terminator_te1550)
        y1 = core.ComponentInstance(dev.ebeam_y_1550)
        gc1 = core.ComponentInstance(dev.ebeam_gc_te1550)
        c1 = [bdc1, bdc1, bdc1, bdc1]
        p1 = [0, 1, 2, 3]
        c2 = [y1, y1, term1, gc1]
        p2 = [0, 1, 0, 0]
        data = zip(c1, p1, c2, p2)
        
        nl = core.Netlist()
        nl.load(data, formatter='ll')
        # TODO: Figure out what the actually correct connections are in the
        # netlist and verify them.

    def test_Netlist_parameterized_initialization(self):
        self.nl = core.Netlist(components=self.components)
        assert self.nl.net_count == 4
        assert len(self.nl.components) == len(self.components)
        # TODO: Figure out what the actually correct connections are in the
        # netlist and verify them.

    def test_Netlist_unparameterized_initialization(self):
        self.nl = core.Netlist()
        for i in range(len(self.components)):
            self.nl.add_component(self.components[i])
        assert len(self.nl.components) == len(self.components)
        # TODO: Figure out what the actually correct connections are in the
        # netlist and verify them.

#     def test_Netlist_externals(self):
#         self.nl = core.Netlist(components=self.components)
#         expected = [comp for comp in self.components if any(x < 0 for x in comp.nets)]
#         actual = self.nl.get_external_components()
#         assert len(expected) == len(actual)
#         for item in expected:
#             assert item in actual
