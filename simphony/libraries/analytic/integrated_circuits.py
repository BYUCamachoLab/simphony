from simphony.circuit import SteadyStateSystem

class VoltageFollower(SteadyStateSystem):
    electrical_ports = ["e0", "e1"]

class OpAmp(SteadyStateSystem):
    electrical_ports = ["ninv","inv","vp","vn","vout"]
    