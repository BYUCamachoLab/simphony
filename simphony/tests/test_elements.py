# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt).
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# This file is part of Simphony.
#
# Simphony is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simphony is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simphony. If not, see <https://www.gnu.org/licenses/>.

import pytest

import simphony.library.ebeam as ebeam


class TestNodes:
    def test_rename(self):
        wg = ebeam.ebeam_wg_integral_1550(50e-6)

        # with pytest.raises(ValueError):
        #     wg._node_idx_by_name('n3')
        # wg.rename_nodes(('n1', 'n3'))
        # assert wg._node_idx_by_name('n3') == 1
