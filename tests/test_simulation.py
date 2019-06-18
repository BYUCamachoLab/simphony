import numpy as np
import pytest

import simphony.core as core
import simphony.DeviceLibrary.devices as dev
import simphony.errors as errors
import simphony.simulation as sim


class Test:
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
        y1 = core.ComponentInstance(cls.y, [4, 0, 1])
        dc1 = core.ComponentInstance(cls.dc, [3, -2])
        wg1 = core.ComponentInstance(cls.wg, [4,-1], extras={'length':40})
        cls.components = [bdc1, term1, y1, dc1, wg1]
        cls.nl = core.Netlist(components=cls.components)
        
    def test_Simulation_initialization(self):
        length = 10
        s = sim.Simulation(None, points=length)
        assert len(s.freq_array == length)

    def test_Simulation_no_initialization(self):
        pass
    
    def test_caching(self):
        simu = sim.Simulation()
        assert np.array_equal(simu.freq_array, np.linspace(1.88e+14, 1.99e+14, 2000))
        simu.netlist = self.nl
        simu.cache_models()
        assert len(simu._cached) == len([item for item in self.components if item.model.cachable])

    def test_Simulation_MatchPorts(self):
        core.clear_models()

        bdc = dev.ebeam_bdc_te1550()
        dc = dev.ebeam_dc_halfring_te1550()
        gc = dev.ebeam_gc_te1550()
        term = dev.ebeam_terminator_te1550()
        wg = dev.ebeam_wg_integral_1550()
        y = dev.ebeam_y_1550()

        bdc1 = core.ComponentInstance(bdc, [0, 1, 2, 3])
        term1 = core.ComponentInstance(term, [2])
        y1 = core.ComponentInstance(y, [-1, 0, 1])
        dc1 = core.ComponentInstance(dc, [3, -2])
        components = [bdc1, term1, y1, dc1]

        nl = core.Netlist(components=components)
        c1, n1, c2, n2 = sim.match_ports(3, nl.components)
        assert c1 == 0
        assert n1 == 3
        assert c2 == 3
        assert n2 == 0

    def test_Simulation_cascade(self):
        simu = sim.Simulation(self.nl)
        simu.cascade()
        assert simu.combined.nets == [-1, -2]
        assert len(simu.combined.f) == 2000
        assert len(simu.combined.s) == 2000
        assert len(simu.combined.nets) == 2
        assert simu.combined.s.shape[1] == 2

    def test_Simulation_rearrange_order(self):
        ports = [-4, -1, -3, -5, -2]
        concat_order = sim.Simulation._rearrange_order(ports)
        print(concat_order)
        assert concat_order == [1, 4, 2, 0, 3]

    def test_Simulation_rearrange(self):
        simu = sim.Simulation(self.nl)
        simu.cascade()

def test_scripting():
    core.clear_models()

    bdc = dev.ebeam_bdc_te1550()
    dc = dev.ebeam_dc_halfring_te1550()
    gc = dev.ebeam_gc_te1550()
    term = dev.ebeam_terminator_te1550()
    wg = dev.ebeam_wg_integral_1550()
    y = dev.ebeam_y_1550()

    y.get_s_parameters()

    bdc1 = core.ComponentInstance(bdc, [0, 1, 2, 3])
    term1 = core.ComponentInstance(term, [2])
    y1 = core.ComponentInstance(y, [-1, 0, 1])
    dc1 = core.ComponentInstance(dc, [3, -2])
    components = [bdc1, term1, y1, dc1]

    nl = core.Netlist(components=components)
    assert nl.net_count == 4
    simu = sim.Simulation(nl)
    simu.cache_models()
    assert len(simu._cached) == 4

    good = core.ComponentModel("arbitrary_component", [0, 0, 0], cachable=True)
    assert good.get_s_parameters() == [0, 0, 0]
    with pytest.raises(errors.DuplicateModelError):
        error = core.ComponentModel("arbitrary_component", [0, 1, 2])
