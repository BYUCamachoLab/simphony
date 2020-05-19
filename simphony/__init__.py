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

"""
simphony
========

A Simulator for Photonic circuits
"""

import platform
import sys

if sys.version_info < (3, 0, 0):
    raise Exception(
        "Simphony requires Python 3 (version "
        + platform.python_version()
        + " detected)."
    )

__version__ = "0.3.0"
__license__ = __doc__
__project_url__ = "https://github.com/BYUCamachoLab/simphony"
__forum_url__ = "https://github.com/BYUCamachoLab/simphony/issues"
__trouble_url__ = __project_url__ + "/wiki/Troubleshooting-Guide"
__website_url__ = "https://camacholab.byu.edu/"
