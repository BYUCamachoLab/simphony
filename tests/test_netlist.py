import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.devices as dev
import pytest

class Test_Netlist:
    @classmethod
    def setup(cls):
        core.clear_models()

        cls.bdc = dev.ebeam_bdc_te1550()
        cls.dc = dev.ebeam_dc_halfring_te1550()
        cls.gc = dev.ebeam_gc_te1550()
        cls.term = dev.ebeam_terminator_te1550()
        cls.wg = dev.ebeam_wg_integral_1550()
        cls.y = dev.ebeam_y_1550()

        bdc1 = core.ComponentInstance(cls.bdc, [0, 1, 2, 3])
        term1 = core.ComponentInstance(cls.term, [2])
        y1 = core.ComponentInstance(cls.y, [-1, 0, 1])
        dc1 = core.ComponentInstance(cls.dc, [3, -2])
        cls.components = [bdc1, term1, y1, dc1]

    def test_parsing_listoflists(self):
        bdc1 = core.ComponentInstance(self.bdc)
        term1 = core.ComponentInstance(self.term)
        y1 = core.ComponentInstance(self.y)
        gc1 = core.ComponentInstance(self.gc)
        c1 = [bdc1, bdc1, bdc1, bdc1]
        p1 = [0, 1, 2, 3]
        c2 = [y1, y1, term1, gc1]
        p2 = [0, 1, 0, 0]
        data = zip(c1, p1, c2, p2)
        
        nl = core.Netlist()
        nl.load(data, formatter='ll')

#     def test_Netlist_parameterized_initialization(self):
#         self.nl = core.Netlist(components=self.components)
#         assert self.nl.net_count == 4
#         assert len(self.nl.components) == len(self.components)

#     def test_Netlist_unparameterized_initialization(self):
#         self.nl = core.Netlist()
#         for i in range(len(self.components)):
#             self.nl.add_component(self.components[i])
#         assert len(self.nl.components) == len(self.components)

#     def test_Netlist_externals(self):
#         self.nl = core.Netlist(components=self.components)
#         expected = [comp for comp in self.components if any(x < 0 for x in comp.nets)]
#         actual = self.nl.get_external_components()
#         assert len(expected) == len(actual)
#         for item in expected:
#             assert item in actual

# # class TestStrToSci(object):
# #     def test_milli(self):
# #         str1 = '3m'
# #         num1 = 3e-3
# #         assert num1 == netlist.strToSci(str1)
# #         str2 = '4.7m'
# #         num2 = 4.7e-3
# #         assert num2 == netlist.strToSci(str2)
# #         str3 = '0.5m'
# #         num3 = 0.5e-3
# #         assert num3 == netlist.strToSci(str3)
# #         str4 = '-0.37m'
# #         num4 = -0.37e-3
# #         assert num4 == netlist.strToSci(str4)
# #         str5 = '-14.3m'
# #         num5 = -14.3e-3
# #         assert num5 == netlist.strToSci(str5)

# #     def test_micro(self):
# #         str1 = '3u'
# #         num1 = 3e-6
# #         assert num1 == netlist.strToSci(str1)
# #         str2 = '4.7u'
# #         num2 = 4.7e-6
# #         assert num2 == netlist.strToSci(str2)
# #         str3 = '0.5u'
# #         num3 = 0.5e-6
# #         assert num3 == netlist.strToSci(str3)
# #         str4 = '-0.37u'
# #         num4 = -0.37e-6
# #         assert num4 == netlist.strToSci(str4)
# #         str5 = '-14.3u'
# #         num5 = -14.3e-6
# #         assert num5 == netlist.strToSci(str5)
    
# #     def test_nano(self):
# #         str1 = '3n'
# #         num1 = 3e-9
# #         assert num1 == pytest.approx(netlist.strToSci(str1))
# #         str2 = '4.7n'
# #         num2 = 4.7e-9
# #         assert num2 == pytest.approx(netlist.strToSci(str2))
# #         str3 = '0.5n'
# #         num3 = 0.5e-9
# #         assert num3 == pytest.approx(netlist.strToSci(str3))
# #         str4 = '-0.37n'
# #         num4 = -0.37e-9
# #         assert num4 == pytest.approx(netlist.strToSci(str4))
# #         str5 = '-14.3n'
# #         num5 = -14.3e-9
# #         assert num5 == pytest.approx(netlist.strToSci(str5))