import numpy as np

from simphony.models import Model, Laser, Detector
from simphony.circuit import Circuit
from simphony.simulation import Simulation


class GratingCoupler(Model):
    onames = ["fa", "wg"]

    def __init__(self, k):
        self.k = k

    def s_params(self, wl):
        print(f"cache miss ({self.k})")
        return wl * 1j * self.k
    

class YJunction(Model):
    onames = ["in", "o1", "o2"]

    def __init__(self, k):
        self.k = k

    def s_params(self, wl):
        print(f"cache miss ({self.k})")
        return wl * 1j * self.k


class Waveguide(Model):
    ocount = 2

    def __init__(self, a):
        self.a = a

    def s_params(self, wl):
        print(f"cache miss ({self.a})")
        return 1j * self.a


class Heater(Model):
    jit = False
    ecount = 2

    def __init__(self, onames=["o0", "o1"]):
        self.onames = onames

    def s_params(self, wl):
        pass


gc_in = GratingCoupler()
y_split = YJunction()
wg_short = Waveguide(length=25)
wg_long = Waveguide(length=50)
y_combine = YJunction()
gc_out = GratingCoupler()

cir = Circuit()
cir.connect(gc_in.o(1), y_split)
cir.connect(y_split, wg_short)
cir.connect(y_split, wg_long)
cir.connect(wg_short, y_combine.o(1))
cir.connect(wg_long, y_combine.o(2))
cir.connect(y_combine, gc_out.o(1))

cir.connect(gc_in.o(0), Laser())
cir.connect(gc_out.o(0), Detector())


def mzi_factory(wg_length=50):
    gc_in = GratingCoupler()
    y_split = YJunction()
    wg_short = Waveguide(length=25)
    wg_long = Waveguide(length=wg_length)
    y_combine = YJunction()
    gc_out = GratingCoupler()

    cir = Circuit()
    cir.connect(gc_in.o(1), y_split)
    cir.connect(y_split, wg_short)
    cir.connect(y_split, wg_long)
    cir.connect(wg_short, y_combine.o(1))
    cir.connect(wg_long, y_combine.o(2))
    cir.connect(y_combine, gc_out.o(1))

    cir.connect(gc_in.o(0), Laser())
    cir.connect(gc_out.o(0), Detector())
    return cir


freq = np.linspace(1.5, 1.6, 2000)
sim = Simulation(cir, freq, context={})
sim = IdealSimulation
sim = SamplingSimulation
sim = TimeDomainSimulation
