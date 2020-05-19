import cmath as cm
import os

import numpy as np


def _read_s_parameters():
    """Returns the s-parameters across some frequency range for the
    ebeam_y_1550 model in the form [frequency, s-parameters].

    Parameters
    ----------
    numports : int
        The number of ports the photonic component has.
    """
    numports = 2
    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sparams",
        "ebeam_gc_te1550",
        "GC_TE1550_thickness=220 deltaw=0.txt",
    )
    with open(filename) as fid:
        # grating coupler compact models have 100 points for each s-matrix index
        arrlen = 100

        lines = fid.readlines()
        F = np.zeros(arrlen)
        S = np.zeros((arrlen, 2, 2), "complex128")
        for i in range(0, arrlen):
            words = lines[i].split()
            F[i] = float(words[0])
            S[i, 0, 0] = cm.rect(float(words[1]), float(words[2]))
            S[i, 0, 1] = cm.rect(float(words[3]), float(words[4]))
            S[i, 1, 0] = cm.rect(float(words[5]), float(words[6]))
            S[i, 1, 1] = cm.rect(float(words[7]), float(words[8]))
        F = F[::-1]
        S = S[::-1, :, :]
    return (F, S)
