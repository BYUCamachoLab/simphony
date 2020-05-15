import matplotlib.pyplot as plt
import numpy as np


def plot_sparameters(c, wav=None):
    """ plots sparameters from a model

    .. plot::
        :include-source:

        from simphony.library import siepic
        from simphony.library.gdsfactory import plot_sparameters

        coupler = siepic.ebeam_dc_halfring_straight()
        plot_sparameters(coupler)
    """
    wav = wav or np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)

    plt.plot(wav, np.abs(s[:, 1] ** 2))
    plt.show()


if __name__ == "__main__":
    from simphony.library import siepic

    coupler = siepic.ebeam_dc_halfring_straight()
    plot_sparameters(coupler)
