# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.die import Die
from simphony.libraries import siepic

@pytest.fixture
def mzi():
    gc_input = siepic.GratingCoupler()
    y_splitter = siepic.YBranch()
    wg_long = siepic.Waveguide(length=150e-6)
    wg_short = siepic.Waveguide(length=50e-6)
    y_recombiner = siepic.YBranch()
    gc_output = siepic.GratingCoupler()

    y_splitter.multiconnect(gc_input, wg_long, wg_short)
    y_recombiner.multiconnect(gc_output, wg_short, wg_long)

    return (gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short)


class TestDie:
    def test_add_components(self, mzi):
        die = Die()
        gc_input, _, _, _, _, _ = mzi

        die.add_components(gc_input.circuit._get_components())

        assert len(die.device_list) == 6
        assert len(die.device_grid.references) == 6
        assert len(die.device_grid_refs) == 6

    def test_connections(self):
        die = Die()
        gc_input,  y_splitter, wg_long, y_recombiner, gc_output, wg_short = mzi

        die.add_components(gc_input.circuit._get_components())

        die.distribute_devices(direction='grid', shape=(3,2), spacing=(5,10))

        y_splitter["pin1"].connect(gc_input["pin1"])

        y_recombiner["pin1"].connect(gc_output["pin1"])

        y_splitter["pin2"].connect(wg_long)
        y_recombiner["pin3"].connect(wg_long)

        y_splitter["pin3"].connect(wg_short)
        y_recombiner["pin2"].connect(wg_short)

        assert die.device_grid.references[0].center != gc_input.device.center
        assert die.device_grid.references[1].center == y_splitter.device.center
        
