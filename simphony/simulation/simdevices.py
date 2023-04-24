"""
Module for simulation devices.
"""

class SimDevice:
    """
    Base class for all sources.
    """
    def __init__(self, ckt, port) -> None:
        self.ckt = ckt
        self.port = port

class Laser(SimDevice):
    """
    Laser source.
    """
    def __init__(self, ckt, port, power, wl, mod_function) -> None:
        super().__init__(ckt, port)
        self.power = power
        self.wl = wl
        self.mod_function = mod_function

class Detector(SimDevice):
    """
    Detector.
    """
    def __init__(self, ckt, port, responsivity) -> None:
        super().__init__(ckt, port)
        self.responsivity = responsivity
