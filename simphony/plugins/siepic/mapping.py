# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

def rearg(component, parameters):
    """
    Maps arguments from spice files to the keyword dictionaries accepted
    by the built-in model libraries, discarding unused parameters.

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
    'ebeam_bdc_te1550': {},
    # 'contra_directional_coupler': {},
    'ebeam_dc_halfring_straight': {},
    'ebeam_dc_te1550': {},
    # 'ebeam_disconnected_te1550': {},
    # 'ebeam_disconnected_tm1550': {},
    # 'ebeam_taper_te1550': {},
    'ebeam_terminator_te1550': {},
    # 'ebeam_terminator_tm1550': {},
    'ebeam_gc_te1550': {},
    'ebeam_wg_integral_1550' : {
        'wg_length': 'length',
        'wg_width': 'width',
    },
    'ebeam_y_1550': {},
}
