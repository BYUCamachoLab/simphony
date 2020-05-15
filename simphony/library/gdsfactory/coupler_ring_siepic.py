import pp
from simphony.library import siepic
from simphony.library.gdsfactory import plot_sparameters

if __name__ == "__main__":
    c = siepic.ebeam_dc_halfring_straight(
        gap=200e-9, radius=10e-6, width=500e-9, thickness=220e-9, couple_length=0.0
    )
    c = plot_sparameters(c)
