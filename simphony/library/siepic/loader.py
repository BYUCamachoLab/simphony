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

import xml.etree.ElementTree as ET
from bisect import bisect_left

tree = ET.parse("dc_map.xml")
root = tree.getroot()

associations = {}

for association in root:
    Lc = float(association.find("design/value").text)
    filename = str(association.find("extracted/value").text)
    associations[Lc] = filename


def take_closest(sorted_list, value):
    """Assumes ``sorted_list`` is sorted. Returns closest value to ``value``.

    If two numbers are equally close, return the smallest number.

    References
    ----------
    https://stackoverflow.com/a/12141511/11530613
    """
    pos = bisect_left(associations, Lc)
    if pos == 0:
        return associations[0]
    if pos == len(associations):
        return associations[-1]
    before = associations[pos - 1]
    after = associations[pos]
    if after - Lc < Lc - before:
        return after
    else:
        return before


associations[take_closest(list(associations.keys()), 1.35e-5)]
