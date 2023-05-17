"""
Module for simulation devices.
"""


class SimDevice:
    """
    Base class for all sources.
    """

    def __init__(self, ckt, ports: list) -> None:
        for p in ports:
            if (p not in ckt._oports) and (p not in ckt._eports):
                raise ValueError(f"Port {p} is not available on the circuit.")
        ckt.sim_devices.append(self)
        self.ckt = ckt
        self.ports = ports


class Laser(SimDevice):
    """
    Laser source.
    """

    def __init__(self, ckt, port, power=1, phase=0, mod_function=None) -> None:
        super().__init__(ckt, [port])
        self.power = power
        self.mod_function = mod_function
        self.phase = phase


class Detector(SimDevice):
    """
    Detector.
    """

    def __init__(self, ckt, port, responsivity=1.0) -> None:
        super().__init__(ckt, [port])
        self.responsivity = responsivity
