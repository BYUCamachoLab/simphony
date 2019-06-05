import simphony.netlist as netlist
import simphony.models.components as components
import pytest

class TestComponentsInNetlist(object):
    def test_abstract_component(self):
        nets = [-1, 3, 4, 8]
        xpos = 0.0554
        ypos = -14.2425
        with pytest.raises(TypeError):
            comp1 = components.Component(nets=nets, lay_x=xpos, lay_y=ypos)
        obj1 = netlist.ObjectModelNetlist()
        assert obj1.component_list == []
        assert obj1.net_count == 0

    def test_ebeam_wg_integral_1550(self):
        nets = [-1, 3, 4, 8]
        xpos = 0.0554
        ypos = -14.2425
        comp1 = components.ebeam_wg_integral_1550(nets=nets, lay_x=xpos, lay_y=ypos, length=40, width=0.5, height=0.22)
        obj1 = netlist.ObjectModelNetlist()
        assert obj1.component_list == []
        assert obj1.net_count == 0
        obj1.component_list.append(comp1)
        assert len(obj1.component_list) == 1

    def test_ebeam_bdc_te1550(self):
        pass

    def test_ebeam_gc_te1550(self):
        pass

    def test_ebeam_y_1550(self):
        pass

    def test_ebeam_terminator_te1550(self):
        pass

    def test_ebeam_dc_halfring_te1550(self):
        pass

class TestObjectModelNetlist(object):
    def test_single_instantiation(self):
        obj1 = netlist.ObjectModelNetlist()
        assert obj1.component_list == []
        assert obj1.net_count == 0

    def test_double_instantiation(self):
        obj1 = netlist.ObjectModelNetlist()
        assert obj1.component_list == []
        assert obj1.net_count == 0

        obj1.component_list.append(components.ebeam_bdc_te1550())
        obj1.component_list.append(components.ebeam_gc_te1550())
        assert len(obj1.component_list) == 2

        obj2 = netlist.ObjectModelNetlist()
        assert obj2.component_list == []
        assert obj2.net_count == 0

class TestComponentSimulation(object):
    def test_clean_instantiation(self):
        obj1 = netlist.ComponentSimulation()
        assert not hasattr(obj1, 'nets')
        assert not hasattr(obj1, 'f')
        assert not hasattr(obj1, 's')

    def test_component_instantiation(self):
        nets = [-1, 3, 4, 8]
        xpos = 0.0554
        ypos = -14.2425
        comp1 = components.ebeam_bdc_te1550(nets=nets, lay_x=xpos, lay_y=ypos)
        obj1 = netlist.ComponentSimulation(comp1)
        assert obj1.nets == nets
        assert obj1.nets is not nets
        assert hasattr(obj1, 'f')
        assert hasattr(obj1, 's')

class TestStrToSci(object):
    def test_milli(self):
        str1 = '3m'
        num1 = 3e-3
        assert num1 == netlist.strToSci(str1)
        str2 = '4.7m'
        num2 = 4.7e-3
        assert num2 == netlist.strToSci(str2)
        str3 = '0.5m'
        num3 = 0.5e-3
        assert num3 == netlist.strToSci(str3)
        str4 = '-0.37m'
        num4 = -0.37e-3
        assert num4 == netlist.strToSci(str4)
        str5 = '-14.3m'
        num5 = -14.3e-3
        assert num5 == netlist.strToSci(str5)

    def test_micro(self):
        str1 = '3u'
        num1 = 3e-6
        assert num1 == netlist.strToSci(str1)
        str2 = '4.7u'
        num2 = 4.7e-6
        assert num2 == netlist.strToSci(str2)
        str3 = '0.5u'
        num3 = 0.5e-6
        assert num3 == netlist.strToSci(str3)
        str4 = '-0.37u'
        num4 = -0.37e-6
        assert num4 == netlist.strToSci(str4)
        str5 = '-14.3u'
        num5 = -14.3e-6
        assert num5 == netlist.strToSci(str5)
    
    def test_nano(self):
        str1 = '3n'
        num1 = 3e-9
        assert num1 == pytest.approx(netlist.strToSci(str1))
        str2 = '4.7n'
        num2 = 4.7e-9
        assert num2 == pytest.approx(netlist.strToSci(str2))
        str3 = '0.5n'
        num3 = 0.5e-9
        assert num3 == pytest.approx(netlist.strToSci(str3))
        str4 = '-0.37n'
        num4 = -0.37e-9
        assert num4 == pytest.approx(netlist.strToSci(str4))
        str5 = '-14.3n'
        num5 = -14.3e-9
        assert num5 == pytest.approx(netlist.strToSci(str5))