# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import copy
from uuid import uuid4

import pytest
import numpy as np

from .context import simphony
from simphony.simulation import Cache

class CustomObject:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()

class TestCache:
    @classmethod
    def setup(cls):
        cls.cache = Cache()

    def test_cache(self):
        c = self.cache
        uid = uuid4()
        c['a'] = copy.deepcopy(uid)
        assert c.contains('a') == True
        assert c['a'] == uid
        assert c['a'] is not uid

    def test_insert(self):
        a = CustomObject('a')
        b = CustomObject('b')
        c = CustomObject('c')
        d = CustomObject('d')
        e = CustomObject('e')
        values = {
            a : uuid4(),
            b : uuid4(),
            c : uuid4(),
            d : uuid4(),
            e : uuid4(),
        }
        ignored = {
            CustomObject('b'): b,
            CustomObject('d'): d,
        }
        for k, v in values.items():
            self.cache[k] = v
        for k, v in ignored.items():
            self.cache[k] = v
            
        assert len(self.cache.keys()) == len(values) + len(ignored)
        
        for key in self.cache.keys():
            try:
                assert self.cache[key] == values[key]
            except KeyError:
                assert self.cache[key] == values[ignored[key]]

        # combined_dict = values.update(ignored)
        # for key in combined_dict.keys():
        #     assert self.cache.contains(key)
    

# class TestSimulation:
#     @classmethod
#     def setup(cls):
#         bdc1 = core.ComponentInstance(dev.ebeam_bdc_te1550, [0, 1, 2, 3])
#         term1 = core.ComponentInstance(dev.ebeam_terminator_te1550, [2])
#         y1 = core.ComponentInstance(dev.ebeam_y_1550, [4, 0, 1])
#         gc1 = core.ComponentInstance(dev.ebeam_gc_te1550, [3, -2])
#         wg1 = core.ComponentInstance(dev.ebeam_wg_integral_1550, [4,-1], extras={'length':40e-6})
#         cls.components = [bdc1, term1, y1, gc1, wg1]
#         cls.nl = core.Netlist(components=cls.components)
        
#     def test_Simulation_initialization(self):
#         length = 1000
#         s = sim.Simulation(self.nl, num=length)
#         assert len(s.freq_array == length)

# #     def test_Simulation_no_initialization(self):
# #         pass
    
#     def test_caching(self):
#         simu = sim.Simulation(self.nl)
#         assert np.array_equal(simu.freq_array, np.linspace(1.88e+14, 1.99e+14, 2000))
#         assert len(simu.cache) == len([item for item in self.components if item.model.cachable])

#     def test_Simulation_MatchPorts(self):
#         bdc1 = core.ComponentInstance(dev.ebeam_bdc_te1550, [0, 1, 2, 3])
#         term1 = core.ComponentInstance(dev.ebeam_terminator_te1550, [2])
#         y1 = core.ComponentInstance(dev.ebeam_y_1550, [-1, 0, 1])
#         gc1 = core.ComponentInstance(dev.ebeam_gc_te1550, [3, -2])
#         components = [bdc1, term1, y1, gc1]

#         nl = core.Netlist(components=components)
#         c1, n1, c2, n2 = sim.match_ports(3, nl.components)
#         assert c1 == 0
#         assert n1 == 3
#         assert c2 == 3
#         assert n2 == 0

#     def test_Simulation_cascade(self):
#         simu = sim.Simulation(self.nl)
#         simu._cascade()
#         assert simu.combined.nets == [0, 1]
#         assert len(simu.combined.f) == 2000
#         assert len(simu.combined.s) == 2000
#         assert len(simu.combined.nets) == 2
#         assert simu.combined.s.shape[1] == 2

#     def test_Simulation_rearrange_order(self):
#         ports = [-4, -1, -3, -5, -2]
#         concat_order = sim.rearrange_order(ports)
#         print(concat_order)
#         assert concat_order == [1, 4, 2, 0, 3]

#     def test_Simulation_rearrange(self):
#         simu = sim.Simulation(self.nl)
#         simu._cascade()
#         # TODO: Create a benchmark case to see what is actually happening.
