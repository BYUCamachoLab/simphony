# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

import simphony.library.ebeam as ebeam


class TestNodes:
    def test_rename(self):
        wg = ebeam.ebeam_wg_integral_1550(50e-6)

        # with pytest.raises(ValueError):
        #     wg._node_idx_by_name('n3')
        # wg.rename_nodes(('n1', 'n3'))
        # assert wg._node_idx_by_name('n3') == 1
