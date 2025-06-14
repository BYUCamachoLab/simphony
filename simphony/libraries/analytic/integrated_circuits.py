from simphony.circuit import SpectralSystem

class VoltageFollower(SpectralSystem):
    electrical_ports = ["e0", "e1"]

class OpAmp(SpectralSystem):
    electrical_ports = ["ninv","inv","vp","vn","vout"]
    