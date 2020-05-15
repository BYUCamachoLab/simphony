import pp
from simphony.library.gdsfactory import load, sweep_simulation


def coupler_ring(c=pp.c.coupler_ring, **kwargs):
    m = load(c, **kwargs)
    return m


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    wav = np.linspace(1520, 1570, 1024) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)

    plt.plot(wav, np.abs(s[:, 1] ** 2))
    print(c.pins)
    plt.show()
