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
        y1 = core.ComponentInstance(cls.y, [-1, 0, 1])
        dc1 = core.ComponentInstance(cls.dc, [3, -2])
        cls.components = [bdc1, term1, y1, dc1]
        cls.nl = core.Netlist(components=cls.components, net_count=3)
        
    def test_Simulation_initialization(self):
        pass

    def test_Simulation_no_initialization(self):
        pass
    
    def test_caching(self):
        simu = sim.Simulation()
        assert np.array_equal(simu.freq_array, np.linspace(1.88e+14, 1.99e+14, 2000))
        simu.netlist = self.nl
        simu.cache_models()
        assert len(simu.cached) == 4

    def test_scripting(self):
        y = self.y
        y.get_s_parameters()
        with pytest.raises(errors.DuplicateModelError):
            error = dev.ebeam_y_1550()
        y_inst1 = core.ComponentInstance(y, [0, 1, 2], 0, 0, None)
        y_inst2 = core.ComponentInstance(y, [3, 1, 2], 0, 0, None)
