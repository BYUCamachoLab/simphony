# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2019 Sequoia Ploeg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

version_info = (0, 2, 1, 'dev0')

# name = "simphony"
# from simphony._version import __version__

__version__ = '.'.join(map(str, version_info))
__license__ = __doc__
__project_url__ = 'https://github.com/BYUCamachoLab/simphony'
__forum_url__   = 'https://github.com/BYUCamachoLab/simphony/issues'
__trouble_url__ = __project_url__ + '/wiki/Troubleshooting-Guide'
__website_url__ = 'https://camacholab.byu.edu/'


__all__ = [
    'core',
    'simulation',
    # 'DeviceLibrary',
]

from . import *
