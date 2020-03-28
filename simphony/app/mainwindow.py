# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.simulation
===================

This file contains the script for running the main GUI.
"""

import sys

from PyQt5.QtWidgets import QApplication

from simphony.app.models.model import Model
from simphony.app.controllers.main_ctrl import MainController
from simphony.app.views.main_view import MainView

class App(QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self.model = Model()
        self.main_controller = MainController(self.model)
        self.main_view = MainView(self.model, self.main_controller)
        self.main_view.show()

def start_gui():
    app = App(sys.argv)
    sys.exit(app.exec_())