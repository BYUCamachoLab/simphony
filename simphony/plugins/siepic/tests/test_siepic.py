# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import os

import pytest

from simphony.plugins.siepic.parser import load_spi

# ==============================================================================
# Test the parser
# ==============================================================================

EBeam_sequoiap_A_v2_result = {
    "circuits": [
        {
            "name": "EBeam_sequoiap_A_v2",
            "ports": ["ebeam_gc_te1550$1_laser", "ebeam_gc_te1550$1_detector1"],
            "subcircuits": "EBeam_sequoiap_A_v2",
            "params": [
                {"name": "sch_x", "value": -1.0},
                {"name": "sch_y", "value": -1.0},
            ],
        }
    ],
    "subcircuits": [
        {
            "name": "EBeam_sequoiap_A_v2",
            "ports": ["ebeam_gc_te1550$1_laser", "ebeam_gc_te1550$1_detector1"],
            "components": [
                {
                    "name": "ebeam_y_1550_67",
                    "model": "ebeam_y_1550",
                    "ports": ["N$80", "N$81", "N$82"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 0.00010077000000000001,
                        "lay_y": 0.00013824,
                        "sch_x": 8.339586207,
                        "sch_y": 11.440551724,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_68",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550$1_laser", "N$80"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 7.687e-05,
                        "lay_y": 0.00013824,
                        "sch_x": 6.361655172,
                        "sch_y": 11.440551724,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_69",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550$1_detector1", "N$83"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 7.687e-05,
                        "lay_y": 1.1240000000000002e-05,
                        "sch_x": 6.361655172,
                        "sch_y": 0.930206897,
                    },
                },
                {
                    "name": "ebeam_y_1550_70",
                    "model": "ebeam_y_1550",
                    "ports": ["N$83", "N$85", "N$84"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 0.00010077000000000001,
                        "lay_y": 1.1240000000000002e-05,
                        "sch_x": 8.339586207,
                        "sch_y": 0.930206897,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_72",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$81", "N$84"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000189995,
                        "wg_width": 5e-07,
                        "points": "[[108.17,140.99],[138.469,140.99],[138.469,8.49],[108.17,8.49]]",
                        "radius": 5.0,
                        "lay_x": 0.000123694,
                        "lay_y": 7.474e-05,
                        "sch_x": 10.236744828,
                        "sch_y": 6.18537931,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_83",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$82", "N$85"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000149995,
                        "wg_width": 5e-07,
                        "points": "[[104.92,389.16],[120.719,389.16],[120.719,267.66],[104.92,267.66]]",
                        "radius": 5.0,
                        "lay_x": 0.000116444,
                        "lay_y": 7.474e-05,
                        "sch_x": 9.636744828,
                        "sch_y": 6.18537931,
                    },
                },
            ],
            "params": {
                "MC_uniformity_width": 0.0,
                "MC_uniformity_thickness": 0.0,
                "MC_resolution_x": 100.0,
                "MC_resolution_y": 100.0,
                "MC_grid": 1e-05,
                "MC_non_uniform": 99.0,
            },
        }
    ],
    "analyses": [
        {
            "definition": {
                "input_unit": "wavelength",
                "input_parameter": "start_and_stop",
            },
            "params": {
                "minimum_loss": 80.0,
                "analysis_type": "scattering_data",
                "multithreading": "user_defined",
                "number_of_threads": 1.0,
                "orthogonal_identifier": 1.0,
                "start": 1.5e-06,
                "stop": 1.6e-06,
                "number_of_points": 3000.0,
                "input": ["EBeam_sequoiap_A_v2,ebeam_gc_te1550$1_detector1"],
                "output": "EBeam_sequoiap_A_v2,ebeam_gc_te1550$1_laser",
            },
        }
    ],
}

MZI4_result = {
    "circuits": [
        {
            "name": "MZI4",
            "ports": ["ebeam_gc_te1550_detector2", "ebeam_gc_te1550_laser1"],
            "subcircuits": "MZI4",
            "params": [
                {"name": "sch_x", "value": -1.0},
                {"name": "sch_y", "value": -1.0},
            ],
        }
    ],
    "subcircuits": [
        {
            "name": "MZI4",
            "ports": ["ebeam_gc_te1550_detector2", "ebeam_gc_te1550_laser1"],
            "components": [
                {
                    "name": "ebeam_y_1550_0",
                    "model": "ebeam_y_1550",
                    "ports": ["N$0", "N$2", "N$1"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 7.4e-06,
                        "lay_y": 0.000127,
                        "sch_x": 0.478534829,
                        "sch_y": 8.212692343,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_1",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_detector2", "N$0"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": -1.6500000000000005e-05,
                        "lay_y": 0.000127,
                        "sch_x": -1.067003336,
                        "sch_y": 8.212692343,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_2",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_laser1", "N$3"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": -1.6500000000000005e-05,
                        "lay_y": 0.000254,
                        "sch_x": -1.067003336,
                        "sch_y": 16.425384686,
                    },
                },
                {
                    "name": "ebeam_y_1550_3",
                    "model": "ebeam_y_1550",
                    "ports": ["N$6", "N$5", "N$4"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 8.993e-05,
                        "lay_y": 0.000127,
                        "sch_x": 5.815491515,
                        "sch_y": 8.212692343,
                        "sch_f": "true",
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_4",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$1", "N$4"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 6.773e-05,
                        "wg_width": 5e-07,
                        "points": "[[14.8,124.25],[82.53,124.25]]",
                        "radius": 5.0,
                        "lay_x": 4.866500000000001e-05,
                        "lay_y": 0.00012425,
                        "sch_x": 3.147013172,
                        "sch_y": 8.034858453,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_5",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$2", "N$5"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000297394,
                        "wg_width": 5e-07,
                        "points": "[[14.8,129.75],[28.64,129.75],[28.64,247.68],[75.36,247.68],[75.36,129.75],[82.53,129.75]]",
                        "radius": 5.0,
                        "lay_x": 4.866500000000001e-05,
                        "lay_y": 0.000188715,
                        "sch_x": 3.147013172,
                        "sch_y": 12.203608153,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_6",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$6", "N$3"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000256152,
                        "wg_width": 5e-07,
                        "points": "[[97.33,127.0],[114.79,127.0],[114.79,254.0],[0.0,254.0]]",
                        "radius": 5.0,
                        "lay_x": 5.777e-05,
                        "lay_y": 0.0001905,
                        "sch_x": 3.735805013,
                        "sch_y": 12.319038514,
                    },
                },
            ],
            "params": {
                "MC_uniformity_width": 0.0,
                "MC_uniformity_thickness": 0.0,
                "MC_resolution_x": 100.0,
                "MC_resolution_y": 100.0,
                "MC_grid": 1e-05,
                "MC_non_uniform": 99.0,
            },
        }
    ],
    "analyses": [
        {
            "definition": {
                "input_unit": "wavelength",
                "input_parameter": "start_and_stop",
            },
            "params": {
                "minimum_loss": 80.0,
                "analysis_type": "scattering_data",
                "multithreading": "user_defined",
                "number_of_threads": 1.0,
                "orthogonal_identifier": 1.0,
                "start": 1.5e-06,
                "stop": 1.6e-06,
                "number_of_points": 2000.0,
                "input": ["MZI4,ebeam_gc_te1550_detector2"],
                "output": "MZI4,ebeam_gc_te1550_laser1",
            },
        }
    ],
}

top_result = {
    "circuits": [
        {
            "name": "top",
            "ports": [
                "ebeam_gc_te1550_laser1",
                "ebeam_gc_te1550_detector2",
                "ebeam_gc_te1550_detector4",
                "ebeam_gc_te1550_detector3",
            ],
            "subcircuits": "top",
            "params": [
                {"name": "sch_x", "value": -1.0},
                {"name": "sch_y", "value": -1.0},
            ],
        }
    ],
    "subcircuits": [
        {
            "name": "top",
            "ports": [
                "ebeam_gc_te1550_laser1",
                "ebeam_gc_te1550_detector2",
                "ebeam_gc_te1550_detector4",
                "ebeam_gc_te1550_detector3",
            ],
            "components": [
                {
                    "name": "ebeam_dc_te1550_0",
                    "model": "ebeam_dc_te1550",
                    "ports": ["N$0", "N$1", "N$3", "N$2"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_width": 5e-07,
                        "gap": 2e-07,
                        "radius": 5e-06,
                        "Lc": 1.5e-05,
                        "lay_x": 2.36e-06,
                        "lay_y": 1.2e-07,
                        "sch_x": 0.082235221,
                        "sch_y": 0.004181452,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_1",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_laser1", "N$4"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": -0.00013533,
                        "lay_y": 1.475e-05,
                        "sch_x": -4.715632378,
                        "sch_y": 0.513970129,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_2",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_detector2", "N$5"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": -0.00012984,
                        "lay_y": -7.662e-05,
                        "sch_x": -4.524330954,
                        "sch_y": -2.669857037,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_3",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_detector4", "N$6"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 9.456e-05,
                        "lay_y": -8.471e-05,
                        "sch_x": 3.294984096,
                        "sch_y": -2.951756586,
                        "sch_r": 180.0,
                    },
                },
                {
                    "name": "ebeam_gc_te1550_4",
                    "model": "ebeam_gc_te1550",
                    "ports": ["ebeam_gc_te1550_detector3", "N$7"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 0.00013005,
                        "lay_y": 3.253e-05,
                        "sch_x": 4.531648495,
                        "sch_y": 1.133521919,
                        "sch_r": 180.0,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_5",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$0", "N$5"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000173487,
                        "wg_width": 5e-07,
                        "points": "[[-11.14,-2.23],[-40.45,-2.23],[-40.45,-76.62],[-113.34,-76.62]]",
                        "radius": 5.0,
                        "lay_x": -6.224e-05,
                        "lay_y": -3.9425e-05,
                        "sch_x": -2.168779718,
                        "sch_y": -1.373781176,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_6",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$4", "N$1"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000116867,
                        "wg_width": 5e-07,
                        "points": "[[-118.83,14.75],[-26.47,14.75],[-26.47,2.47],[-11.14,2.47]]",
                        "radius": 5.0,
                        "lay_x": -6.4985e-05,
                        "lay_y": 8.61e-06,
                        "sch_x": -2.26443043,
                        "sch_y": 0.300019174,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_7",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$8", "N$2"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 7.4217e-05,
                        "wg_width": 5e-07,
                        "points": "[[65.87,29.78],[36.16,29.78],[36.16,2.47],[15.86,2.47]]",
                        "radius": 5.0,
                        "lay_x": 4.0865e-05,
                        "lay_y": 1.6125e-05,
                        "sch_x": 1.423958598,
                        "sch_y": 0.561882599,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_8",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$3", "N$6"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 0.000141577,
                        "wg_width": 5e-07,
                        "points": "[[15.86,-2.23],[35.04,-2.23],[35.04,-84.71],[78.06,-84.71]]",
                        "radius": 5.0,
                        "lay_x": 4.696e-05,
                        "lay_y": -4.347000000000001e-05,
                        "sch_x": 1.636341509,
                        "sch_y": -1.51473095,
                    },
                },
                {
                    "name": "ebeam_y_1550_9",
                    "model": "ebeam_y_1550",
                    "ports": ["N$8", "N$10", "N$9"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 7.327e-05,
                        "lay_y": 2.978e-05,
                        "sch_x": 2.553124838,
                        "sch_y": 1.037696979,
                    },
                },
                {
                    "name": "ebeam_terminator_te1550_10",
                    "model": "ebeam_terminator_te1550",
                    "ports": ["N$11"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "lay_x": 9.14e-05,
                        "lay_y": 2.7e-07,
                        "sch_x": 3.184872529,
                        "sch_y": 0.009408267,
                        "sch_r": 270.0,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_11",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$9", "N$11"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 3.0488e-05,
                        "wg_width": 5e-07,
                        "points": "[[80.67,27.03],[91.4,27.03],[91.4,5.72]]",
                        "radius": 5.0,
                        "lay_x": 8.641e-05,
                        "lay_y": 1.675e-05,
                        "sch_x": 3.010993821,
                        "sch_y": 0.5836609940000002,
                    },
                },
                {
                    "name": "ebeam_wg_integral_1550_12",
                    "model": "ebeam_wg_integral_1550",
                    "ports": ["N$10", "N$7"],
                    "params": {
                        "library": "Design kits/ebeam",
                        "wg_length": 3.288e-05,
                        "wg_width": 5e-07,
                        "points": "[[80.67,32.53],[113.55,32.53]]",
                        "radius": 5.0,
                        "lay_x": 9.711e-05,
                        "lay_y": 3.253e-05,
                        "sch_x": 3.383839949,
                        "sch_y": 1.133521919,
                    },
                },
            ],
            "params": {
                "MC_uniformity_width": 0.0,
                "MC_uniformity_thickness": 0.0,
                "MC_resolution_x": 100.0,
                "MC_resolution_y": 100.0,
                "MC_grid": 1e-05,
                "MC_non_uniform": 99.0,
            },
        }
    ],
    "analyses": [
        {
            "definition": {
                "input_unit": "wavelength",
                "input_parameter": "start_and_stop",
            },
            "params": {
                "minimum_loss": 80.0,
                "analysis_type": "scattering_data",
                "multithreading": "user_defined",
                "number_of_threads": 1.0,
                "orthogonal_identifier": 1.0,
                "start": 1.5e-06,
                "stop": 1.6e-06,
                "number_of_points": 2000.0,
                "input": [
                    "top,ebeam_gc_te1550_detector2",
                    "top,ebeam_gc_te1550_detector3",
                    "top,ebeam_gc_te1550_detector4",
                ],
                "output": "top,ebeam_gc_te1550_laser1",
            },
        }
    ],
}


def test_EBeam_sequoiap_A_v2():
    filename = os.path.join(
        os.path.dirname(__file__),
        "spice",
        "EBeam_sequoiap_A_v2",
        "EBeam_sequoiap_A_v2_main.spi",
    )
    res = load_spi(filename)
    assert res == EBeam_sequoiap_A_v2_result


def test_MZI4():
    filename = os.path.join(os.path.dirname(__file__), "spice", "MZI4", "MZI4_main.spi")
    res = load_spi(filename)
    assert res == MZI4_result


def test_top():
    filename = os.path.join(os.path.dirname(__file__), "spice", "top", "top_main.spi")
    res = load_spi(filename)
    assert res == top_result


# ==============================================================================
# Test the builder
# ==============================================================================
# import os
# filename = os.path.join('tests', 'spice', 'MZI4', 'MZI4_main.spi')
# filename = os.path.join('tests', 'spice', 'EBeam_sequoiap_A_v2', 'EBeam_sequoiap_A_v2_main.spi')
# filename = os.path.join('tests', 'spice', 'top', 'top_main.spi')
# data = load_spi(filename)
# from simphony.plugins.siepic.builders import build_circuit
# build_circuit(data, 'simphony.library.siepic')
