from simphony.libraries.ideal import coupler, waveguide


class TestCoupler:
    def test_instantiable(self):
        coupler()
        coupler(coupling=0.5)
        coupler(coupling=0.5, loss=0.1)
        coupler(coupling=0.5, loss=0.1, phi=0.1)

    # def test_s_params(self, std_wl_um):
    #     c = coupler(coupling=0.5, loss=0.1)
    #     s = c.s_params(std_wl_um)
    #     c = coupler(coupling=0.5, loss=(0.1, 0.1, 0.1, 0.1))
    #     s = c.s_params(std_wl_um)
    #     c = coupler(coupling=0.5, loss=0.1, phi=0.1)
    #     s = c.s_params(std_wl_um)


class TestWaveguide:
    def test_instantiable(self):
        waveguide(length=100)
        waveguide(length=100, loss=0.1)
        waveguide(length=100, neff=2.3)
        waveguide(length=100, ng=3.4)
        waveguide(length=100, neff=2.3, ng=3.4)
        waveguide(length=100, neff=2.3, ng=3.4, loss=0.1)
        waveguide(length=100, wl0=1.35, neff=2.3, ng=3.4, loss=0.1)

    # def test_s_params(self, std_wl_um):
    #     w = waveguide(length=100)
    #     s = w.s_params(std_wl_um)
    #     w = waveguide(length=100, loss=0.1)
    #     s = w.s_params(std_wl_um)
    #     w = waveguide(length=100, neff=2.3)
    #     s = w.s_params(std_wl_um)
    #     w = waveguide(length=100, ng=3.4)
    #     s = w.s_params(std_wl_um)
    #     w = waveguide(length=100, wl0=1.35)
    #     s = w.s_params(std_wl_um)
