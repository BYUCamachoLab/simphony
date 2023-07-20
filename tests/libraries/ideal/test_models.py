import pytest


from simphony.libraries.ideal import Coupler, Waveguide, PhaseShifter, Terminator


class TestCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Coupler(coupling=1.1)
        with pytest.raises(ValueError):
            Coupler(coupling=-0.1)
        with pytest.raises(ValueError):
            Coupler(coupling=0.5, loss=-0.1)
        with pytest.raises(ValueError):
            Coupler(coupling=0.5, loss=[0.1] * 4)

    def test_instantiable(self):
        Coupler()
        Coupler(coupling=0.5)
        Coupler(coupling=0.5, loss=0.1)
        Coupler(coupling=0.5, loss=(0.1, 0.1, 0.1, 0.1))
        Coupler(coupling=0.5, loss=0.1, phi=0.1)

    def test_s_params(self, std_wl_um):
        c = Coupler(coupling=0.5, loss=0.1)
        s = c.s_params(std_wl_um)
        c = Coupler(coupling=0.5, loss=(0.1, 0.1, 0.1, 0.1))
        s = c.s_params(std_wl_um)
        c = Coupler(coupling=0.5, loss=0.1, phi=0.1)
        s = c.s_params(std_wl_um)


class TestWaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(TypeError):
            Waveguide()

    def test_instantiable(self):
        Waveguide(length=100)
        Waveguide(length=100, loss=0.1)
        Waveguide(length=100, neff=2.3)
        Waveguide(length=100, ng=3.4)
        Waveguide(length=100, neff=2.3, ng=3.4)
        Waveguide(length=100, neff=2.3, ng=3.4, loss=0.1)
        Waveguide(length=100, wl0=1.35, neff=2.3, ng=3.4, loss=0.1)

    def test_s_params(self, std_wl_um):
        w = Waveguide(length=100)
        s = w.s_params(std_wl_um)
        w = Waveguide(length=100, loss=0.1)
        s = w.s_params(std_wl_um)
        w = Waveguide(length=100, neff=2.3)
        s = w.s_params(std_wl_um)
        w = Waveguide(length=100, ng=3.4)
        s = w.s_params(std_wl_um)
        w = Waveguide(length=100, wl0=1.35)
        s = w.s_params(std_wl_um)


class TestPhaseShifter:
    def test_instantiate(self):
        PhaseShifter()
        PhaseShifter(phase=0.1)
        PhaseShifter(phase=0.1, loss=0.1)

    def test_s_params(self, std_wl_um):
        p = PhaseShifter()
        s = p.s_params(std_wl_um)
        p = PhaseShifter(phase=0.1)
        s = p.s_params(std_wl_um)
        p = PhaseShifter(phase=0.1, loss=0.1)
        s = p.s_params(std_wl_um)


class TestTerminator:
    def test_instantiate(self):
        Terminator()
