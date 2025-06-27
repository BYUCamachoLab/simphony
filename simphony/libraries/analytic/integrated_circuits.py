from simphony.circuit import SteadyStateComponent

class VoltageFollower(SteadyStateComponent):
    electrical_ports = ["e0", "e1"]

class OpAmp(SteadyStateComponent):
    electrical_ports = ["ninv","inv","vp","vn","vout"]
    