from simphony.models import Model


class Coupler(Model):
    onames = ["o1", "o2", "o3", "o4"]

    def __init__(self, k):
        self.k = k

    def s_params(self, wl):
        # Fake value
        print(f"cache miss ({self.k})")
        return wl * 1j * self.k


class Waveguide(Model):
    ocount = 2
    ecount = 2

    def __init__(self, a):
        self.a = a

    def s_params(self, wl):
        # Fake value
        print(f"cache miss ({self.a})")
        return 1j * self.a


class Heater(Model):
    jit = False

    def __init__(self, onames=["o0", "o1"]):
        self.onames = onames

    def s_params(self, wl):
        # Fake value
        pass
