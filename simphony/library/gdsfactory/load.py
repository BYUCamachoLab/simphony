import numpy as np
import pp
from scipy.constants import c as SPEED_OF_LIGHT
from simphony.elements import Model
from simphony.tools import freq2wl, interpolate, wl2freq


def load(component, **kwargs):
    """ load Sparameters for a component

    Args:
        component: component factory or instance
        **kwargs
    """
    component = pp.call_if_func(component, **kwargs)
    pins, f, s = pp.sp.load(component)

    def interpolate_sp(freq):
        return interpolate(freq, f, s)

    m = Model()
    m.pins = pins
    m.s_params = (f, s)
    m.s_parameters = interpolate_sp
    m.freq_range = (
        m.s_params[0][0],
        m.s_params[0][-1],
    )  #: The valid frequency range for this model.
    return m


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = load(pp.c.mzi())
    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    plt.plot(wav, np.abs(s[:, 1] ** 2))

    plt.show()
