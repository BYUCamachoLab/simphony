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


def rearg(component, parameters):
    """Maps arguments from spice files to the keyword dictionaries accepted by
    the built-in model libraries, discarding unused parameters.

    Parameters
    ----------
    component : str
    parameters : dict

    Returns
    -------
    args : dict
    """
    mapping = components[component]
    results = {}
    for k, v in parameters.items():
        if k in mapping:
            results[mapping[k]] = v
    return results


components = {
    "ebeam_bdc_te1550": {},
    # 'contra_directional_coupler': {},
    "ebeam_dc_halfring_straight": {},
    "ebeam_dc_te1550": {},
    # 'ebeam_disconnected_te1550': {},
    # 'ebeam_disconnected_tm1550': {},
    # 'ebeam_taper_te1550': {},
    "ebeam_terminator_te1550": {},
    # 'ebeam_terminator_tm1550': {},
    "ebeam_gc_te1550": {},
    "ebeam_wg_integral_1550": {"wg_length": "length", "wg_width": "width",},
    "ebeam_y_1550": {},
}
