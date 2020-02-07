# For file in simphony/app/res that ends in .ui
    # Execute pyuic5 filename -o views/filename_ui.py

import sys
import os
import subprocess

try:
    import PyQt5.uic.pyuic
except ImportError:
    raise ImportError("pyuic5 module could not be found. Aborting...")
    sys.exit()

path = os.path.join('..', 'simphony', 'app')
res = os.path.join(path, 'resources')
views = os.path.join(path, 'views')

for item in os.listdir(res):
    if item.endswith('.ui'):
        name, ext = os.path.splitext(item)
        # subprocess.call(['pyuic5', ])
        print(name, ext)