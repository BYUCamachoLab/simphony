# -*- coding: utf-8 -*-
#
# Copyright © Sequoia Ploeg
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
Tool that converts all .ui files in the simphony.app.resources folder to
python files in simphony.app.views.

Usage:
$ python3 ui2py.py
"""

import sys
import os
import subprocess

try:
    import PyQt5.uic.pyuic
except ImportError:
    raise ImportError("pyuic5 module could not be found. Aborting...")
    sys.exit()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import simphony
os.chdir(os.path.join(simphony.__path__[0], os.pardir))

path = os.path.join('simphony', 'app')
res = os.path.join(path, 'resources')
views = os.path.join(path, 'views')

for item in os.listdir(res):
    if item.endswith('.ui'):
        name, _ = os.path.splitext(item)
        rename = name + '_ui' + '.py'
        path2src = os.path.join(res, item)
        path2dest = os.path.join(views, rename)
        subprocess.call(['python3', '-m', 'PyQt5.uic.pyuic', path2src, '-o', path2dest])
