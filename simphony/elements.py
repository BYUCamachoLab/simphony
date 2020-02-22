# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from simphony import classproperty

class Element:
    __name__ = None
    nodes = tuple()

    @classproperty
    def node_count(cls):
        return len(cls.nodes)

class OpticalElement:
    pass