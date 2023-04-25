"""
Module for simulation devices.
"""

class SimDevice:
    """
    Base class for all sources.
    """
    def __init__(self, ckt, port) -> None:
        if (port not in ckt._oports) and (port not in ckt._eports):
            raise ValueError(f"Port {port} is not available on the circuit.")
        ckt.sim_devices.append(self)
        self.port = port
        self.ckt = ckt
        self.port = port

class Laser(SimDevice):
    """
    Laser source.
    """
    def __init__(self, ckt, port, power=1, mod_function=None) -> None:
        super().__init__(ckt, port)
        self.power = power
        self.mod_function = mod_function        

class Detector(SimDevice):
    """
    Detector.
    """
    def __init__(self, ckt, port, responsivity=1.0) -> None:
        super().__init__(ckt, port)
        self.responsivity = responsivity