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

import os

import numpy as np
import pytest

from simphony.libraries import siepic
from simphony.plugins import interconnect
from simphony.tools import wl2freq


@pytest.fixture(params=["subcircuit", "model"])
def model(request):
    if request.param == "model":
        return siepic.YBranch()
    if request.param == "subcircuit":
        # components
        gc_input = siepic.GratingCoupler()
        y_splitter = siepic.YBranch()
        wg_long = siepic.Waveguide(length=150e-6)
        wg_short = siepic.Waveguide(length=50e-6)
        y_recombiner = siepic.YBranch()
        gc_output = siepic.GratingCoupler()

        # connections
        y_splitter.multiconnect(gc_input, wg_long, wg_short)
        y_recombiner.multiconnect(gc_output, wg_short, wg_long)

        return y_splitter.circuit.to_subcircuit(permanent=False)


def test_export(model):
    # set things up
    num_pts = 2
    wl = np.linspace(1500, 1600, num_pts) * 1e-9
    freq = wl2freq(wl)[::-1]

    # export
    interconnect.export(model, "temp_wl.sparams", wl=wl)
    interconnect.export(model, "temp_freq.sparams", freq=freq)

    num_pins = model.pin_count
    num_headers = num_pins * num_pins

    # check files to make sure they're the same
    with open("temp_wl.sparams", "r") as f_wl:
        f_wl_lines = f_wl.readlines()
    with open("temp_freq.sparams", "r") as f_freq:
        f_freq_lines = f_freq.readlines()
    assert f_wl_lines == f_freq_lines

    # iterate through lines, making sure it's in the correct format
    headers = 0
    pts = 0
    for i in f_wl_lines:
        if i[0] == "(":
            headers += 1
        else:
            pts += 1

    headers /= 2
    pts /= num_headers

    # correct number of headers and pts in each section
    assert headers == num_headers
    assert pts == num_pts

    # remove files
    os.remove("temp_wl.sparams")
    os.remove("temp_freq.sparams")


@pytest.fixture(params=["awg_1x9.dat", "attenuatortetm.dat", "awg_1x2x2x4.dat"])
def file(request):
    return request.param


def test_load(file):
    filename = os.path.join(os.path.dirname(__file__), file)
    interconnect.load(filename)


# Couldn't get this test to work, no easy way to get order of lines exactly the same
# @pytest.fixture(params=["awg_1x9", "awg_1x2x2x4"])
# def cleaned_file(request):
#     return request.param

# def test_interconnect_both(cleaned_file):
#     #load cleaned file normally
#     with open(cleaned_file+"_clean.dat", "r") as f:
#         file_lines = (f.readlines())

#     #import, export, read
#     model = interconnect.load(cleaned_file+".dat")()
#     interconnect.export(model, "temp.dat", freq=model._f)
#     with open("temp.dat", "r") as f:
#         processed_lines = (f.readlines())

#     assert len(processed_lines) == len(file_lines)

#     for i in range(len(file_lines)):
#         if file_lines[i][0] == "(":
#             assert interconnect._parse_header(file_lines[i]) == interconnect._parse_header(processed_lines[i])

#     # os.remove("temp.dat")
