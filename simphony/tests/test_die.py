# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.die import Die
from simphony.libraries import siepic


@pytest.fixture
def mzi():
    gc_input = siepic.GratingCoupler(name="gcinput")
    y_splitter = siepic.YBranch(name="ysplitter")
    wg_long = siepic.Waveguide(length=150e-6, name="wglong")
    wg_short = siepic.Waveguide(length=50e-6, name="wgshort")
    y_recombiner = siepic.YBranch(name="yrecombiner")
    gc_output = siepic.GratingCoupler(name="gcoutput")

    return (gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short)


class TestDie:
    def test_add_components(self, mzi):
        die = Die()
        gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short = mzi

        die.add_components(
            [gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short]
        )

        assert len(die.device_list) == 6
        assert len(die.device_grid.references) == 6
        assert len(die.device_grid_refs) == 6

    def test_connections(self, mzi):
        die = Die()
        gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short = mzi

        die.add_components(
            [gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short]
        )

        die.distribute_devices(direction="grid", shape=(3, 2), spacing=(5, 10))

        y_splitter["pin1"].connect(gc_input["pin1"])

        y_recombiner["pin1"].connect(gc_output["pin1"])

        y_splitter["pin2"].connect(wg_long)
        y_recombiner["pin3"].connect(wg_long)

        y_splitter["pin3"].connect(wg_short)
        y_recombiner["pin2"].connect(wg_short)

        assert not np.allclose(
            die.device_grid.references[0].center, gc_input.device.center
        )
        assert not np.allclose(
            die.device_grid.references[1].center, y_splitter.device.center
        )
        assert not np.allclose(
            die.device_grid.references[3].center, y_recombiner.device.center
        )
        assert not np.allclose(
            die.device_grid.references[4].center, gc_output.device.center
        )

        assert np.allclose(
            die.device_grid.references[0].ports["pin1"].center,
            die.device_grid.references[1].ports["pin1"].center,
        )
        assert np.allclose(
            die.device_grid.references[3].ports["pin1"].center,
            die.device_grid.references[4].ports["pin1"].center,
        )

    def test_distribute_devices(self, mzi):
        die = Die()
        gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short = mzi

        die.add_components(
            [gc_input, y_splitter, wg_long, y_recombiner, gc_output, wg_short]
        )

        die.distribute_devices(direction="grid", shape=(3, 2), spacing=(5, 10))

        for ref1 in die.device_grid.references:
            for ref2 in die.device_grid.references:
                if (
                    ref1 != ref2
                    and ref1.parent.name[:2] not in ("wg", "Un")
                    and ref2.parent.name[:2] not in ("wg", "Un")
                ):
                    print(ref1.parent.name[:2])
                    assert not np.allclose(ref1.center, ref2.center)

    def test_move(self, mzi):
        die = Die()
        gc_input, _, _, _, _, _ = mzi

        die.add_components([gc_input])

        die.move(gc_input, distance=(5, 0))

        assert not np.allclose(die.device_grid.references[0].center, [0.0, 0.0])
