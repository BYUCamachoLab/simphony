# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

# from simphony import classproperty
from simphony.errors import MissingAttributeError

class Element:
    nodes = None

    @classmethod
    def _class_attr_(cls):
        return [a for a in dir(cls) if not a.startswith('__') \
        and not callable(getattr(cls, a)) \
        and not isinstance(getattr(cls, a), property)]

    def __new__(cls, *args, **kwargs):
        missed = [Element._class_attr_()[i] for i, a in enumerate(Element._class_attr_()) if getattr(cls, a) == None]
        if any(missed):
            raise MissingAttributeError(cls, missed)
        return super().__new__(cls)

    # def __init__(self):
    #     # self.__dict__ = {key: getattr(self, key) for key in Element._class_attr_()}
    #     super().__init__()

    def __eq__(self, other):
        # return self.__dict__ == other.__dict__
        diff = set(self.__dict__) ^ set(other.__dict__)
        match = set(self.__dict__.keys()).intersection(set(other.__dict__.keys()))
        for key in diff:
            if key not in self._class_attr_():
                return False
        for key in match:
            if key not in self._class_attr_():
                if self.__dict__[key] != other.__dict__[key]:
                    return False
        return True

    def rename_nodes(self, nodes):
        """
        Renames the nodes for the instance object. Order is preserved and only
        names are remapped.

        Parameters
        ----------
        nodes : tuple
            The string names of the new nodes, in order, as a tuple.
        """
        self.nodes = nodes
        """
        Do we make this a class method? Does renaming nodes do it for an 
        instance, or globally?
        """
        self.nodes = nodes