import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sax

from simphony.libraries import siepic
from simphony.time_domain import QuantumTimeSim
from simphony.utils import dict_to_matrix

netlist = {
    "instances": {
        "wg": "waveguide",
    },
    "connections": {},
    "ports": {
        "o0": "wg,o0",
        "o1": "wg,o1",
    },
}
circuit, info = sax.circuit(
    netlist=netlist,
    models={
        "waveguide": siepic.waveguide,
    },
)

wvl_microns = np.linspace(1.51, 1.59, 200)
center_wvl = 1.55

ckt = circuit(wl=wvl_microns, wg={"length": 18, "loss": 1000})
S = np.asarray(dict_to_matrix(ckt))
plt.plot(wvl_microns, np.abs(S[:, 0, 1]) ** 2)
plt.show()
