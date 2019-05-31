"""Simphony Photonic Simulator

This module implements a free and open source photonic integrated circuit (PIC)
simulation engine. It is speedy and easily extensible.
"""

name = "simphony"
from simphony._version import __version__
__author__ = 'Sequoia Ploeg, Hyrum Gunther'

__all__ = [
    'netlist',
    'simulation',
    'models',
    'settings_gui',
]

from . import *

import atexit
import configparser
from importlib import import_module
import os

def on_open():
    print('Simphony Python Integration (CamachoLab)')
    try:
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "settings.ini"))
        selections = config['MODEL_SELECT']

        mod = import_module('.models', __name__)
        Comp = mod.components.Component
        for class_ in Comp.__subclasses__():
            class_.set_model(selections[class_.__name__])
    except:
        print("Persistent settings could not be read.")

def on_close():
    config = configparser.ConfigParser()
    config['MODEL_SELECT'] = {}
    selections = config['MODEL_SELECT']

    mod = import_module('.models', __name__)
    Comp = mod.components.Component
    for class_ in Comp.__subclasses__():
        selections[class_.__name__] = class_._selected_model
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "settings.ini"), 'w') as configfile:
        config.write(configfile)
    print("Simphony Integration Closed")

on_open()

atexit.register(on_close)