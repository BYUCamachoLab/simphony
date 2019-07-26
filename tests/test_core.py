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

class TestBase:
    pass
    # def test_ComponentModel_duplicity(self):
    #     fake_s_params = ([1500, 1550, 1600], [0,0,0])
    #     with pytest.raises(errors.DuplicateModelError):
    #         rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
    #         rr2 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
    #     with pytest.raises(errors.DuplicateModelError):
    #         rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
    #         rr2 = copy.deepcopy(rr1)

    # def test_ComponentModel_cachable(self):
    #     fake_s_params = ([1500, 1550, 1600], [0,0,0])
    #     rr1 = core.ComponentModel("ring_resonator_1", 4, fake_s_params, cachable=True)
    #     rr2 = core.ComponentModel("ring_resonator_2", 4, fake_s_params, cachable=True)
    #     assert fake_s_params == rr1.get_s_parameters()
    #     assert fake_s_params == rr1.get_s_parameters(fake_keyword=3, faker_keyword="long")
    #     with pytest.raises(ValueError):
    #         rr3 = core.ComponentModel("ring_resonator_3", 4, cachable=True)

    # def test_ComponentModel_uncachable(self):
    #     fake_s_params = ([1500, 1550, 1600], [0,0,0])
    #     rr1 = core.ComponentModel("ring_resonator_1", 4, fake_s_params, cachable=False)
    #     with pytest.raises(NotImplementedError):
    #         rr1.get_s_parameters()
    #     rr2 = core.ComponentModel("ring_resonator_2", 4, fake_s_params, cachable=False)
    #     rr3 = core.ComponentModel("ring_resonator_3", 4, cachable=False)

    # def test_ComponentInstance_cachableModel(self):
    #     fake_s_params = ([1500, 1550, 1600], [0,0,0])
    #     rr1 = core.ComponentModel("ring_resonator", 4, fake_s_params, cachable=True)
    #     ci1 = core.ComponentInstance(rr1, [0,1,2,3], {'extras':'should be ignored'})
    #     assert rr1.get_s_parameters() == fake_s_params
    #     assert rr1.get_s_parameters() == ci1.get_s_parameters()
    #     assert ci1.get_s_parameters() == fake_s_params

    # def test_ComponentInstance_uncachableModel(self):
    #     wg1 = core.ComponentModel("variable_length_waveguide", 2, cachable=False)
    #     def wg_s_parameters(freq, length, height):
    #         # Some random equation just to allow us to see the effects of the function
    #         return height*np.sin(freq)+length
    #     wg1.get_s_parameters = wg_s_parameters

    #     extras = {
    #         'freq': np.linspace(0,2*np.pi),
    #         'length': 2,
    #         'height':0.5
    #     }
    #     ci1 = core.ComponentInstance(wg1, [0,1], extras)
    #     expected = wg_s_parameters(extras['freq'], extras['length'], extras['height'])
    #     assert np.array_equal(expected, ci1.get_s_parameters())
    #     assert np.array_equal(expected, wg1.get_s_parameters(**extras))

    # def test_core(self):
    #     rr = core.ComponentModel("ring_resonator", ([1500, 1550, 1600], [0,0,0]), cachable=True)
    #     nl = core.Netlist()
    #     c1 = core.ComponentInstance(rr, [0,1,2,3], lay_x=3.1, lay_y=4)
    #     c2 = core.ComponentInstance(rr, [4,1,2,5])
    #     nl.add_component(c1)
    #     nl.add_component(c2)
    #     assert len(nl.components) == 2

# class TestNetlist2:
#     @classmethod
#     def setup(cls):
#         bdc1 = core.ComponentInstance(cls.bdc, [0, 1, 2, 3])
#         term1 = core.ComponentInstance(cls.term, [2])
#         y1 = core.ComponentInstance(cls.y, [-1, 0, 1])
#         dc1 = core.ComponentInstance(cls.dc, [3, -2])
#         cls.components = [bdc1, term1, y1, dc1]

#     def test_parsing_listoflists(self):
#         bdc1 = core.ComponentInstance(self.bdc)
#         term1 = core.ComponentInstance(self.term)
#         y1 = core.ComponentInstance(self.y)
#         gc1 = core.ComponentInstance(self.gc)
#         c1 = [bdc1, bdc1, bdc1, bdc1]
#         p1 = [0, 1, 2, 3]
#         c2 = [y1, y1, term1, gc1]
#         p2 = [0, 1, 0, 0]
#         data = zip(c1, p1, c2, p2)
        
#         nl = core.Netlist()
#         nl.load(data, formatter='ll')

#     def test_Netlist_parameterized_initialization(self):
#         self.nl = core.Netlist(components=self.components)
#         assert self.nl.net_count == 4
#         assert len(self.nl.components) == len(self.components)

#     # def test_Netlist_unparameterized_initialization(self):
#     #     self.nl = core.Netlist()
#     #     for i in range(len(self.components)):
#     #         self.nl.add_component(self.components[i])
#     #     assert len(self.nl.components) == len(self.components)

# #     def test_Netlist_externals(self):
# #         self.nl = core.Netlist(components=self.components)
# #         expected = [comp for comp in self.components if any(x < 0 for x in comp.nets)]
# #         actual = self.nl.get_external_components()
# #         assert len(expected) == len(actual)
# #         for item in expected:
# #             assert item in actual
