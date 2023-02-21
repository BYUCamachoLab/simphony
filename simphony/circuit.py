"""
Define circuit and connections in simphony.
"""

from typing import Union

from simphony.models import Model, Port, OPort, EPort


class Circuit:
    """
    Examples
    --------
    .. code-block:: python

        with Circuit() as cir:
            gc = cir.add(GratingCoupler())
            y = YJunction()
            cir.connect(gc.o(1), y.o(1))
    """
    _model_instances = {}
    
    def __init__(self, name):
        self.name = name
        self._components = []
        self._onodes = []
        self._enodes = []

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def add(self, component):
        self._components.append(component)
        return component

    def connect(self, port1: Union[Model, Port], port2: Union[Model, Port]):
        if isinstance(port1, OPort):
            if isinstance(port2, OPort):
                pass
            elif issubclass(type(port2), Model):
                p2 = port2.next_unconnected_oport()

        if isinstance(port1, EPort):
            if isinstance(port2, EPort):
                pass

        if issubclass(type(port1), Model):
            p1 = port1.next_unconnected_oport()

    def disconnect(self, port):
        pass
