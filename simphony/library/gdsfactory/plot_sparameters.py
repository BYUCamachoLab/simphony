import matplotlib.pyplot as plt
import numpy as np


def plot_sparameters(c, wavelengths=None):
    """ plots sparameters from a model

    .. plot::
        :include-source:

        from simphony.library import siepic
        from simphony.library.gdsfactory import plot_sparameters

        coupler = siepic.ebeam_dc_halfring_straight()
        plot_sparameters(coupler)
    """
    if wavelengths is None:
        wavelengths = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wavelengths
    s = c.s_parameters(freq=f)
    # _, i, _ = np.shape(s)
    # plt.plot(wavelengths, np.abs(s[:, 1] ** 2))

    for i, pin in enumerate(c.pins):
        plt.plot(wavelengths * 1e9, 100 * np.abs(s[:, i, 0] ** 2), label=pin)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Transmission (%)")
    return s


if __name__ == "__main__":
    from simphony.library import siepic

    coupler = siepic.ebeam_dc_halfring_straight(
        gap=200e-9, radius=10e-6, width=500e-9, thickness=220e-9, couple_length=0.0
    )
    wavelengths = np.linspace(1510, 1600, 1024) * 1e-9
    plot_sparameters(coupler, wavelengths=wavelengths)
    plt.legend()
    plt.show()
