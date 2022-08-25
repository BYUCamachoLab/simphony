# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
layout_aware.py
---------------

Author: Skandan Chandrasekar
Modified: August 10, 2022

A script that walks the user through the process of running layout-aware Monte Carlo simulations.
"""


# import necessary modules
import gdsfactory as gf
import matplotlib.pyplot as plt

from simphony.libraries import siepic
from simphony.simulation import Detector, Laser, Simulation

# instantiate components
ysplitter = siepic.YBranch(name="ysplitter")
gcinput = siepic.GratingCoupler(name="gcinput")
gcoutput = siepic.GratingCoupler(name="gcoutput")
yrecombiner = siepic.YBranch(name="yrecombiner")
wg_long = siepic.Waveguide(name="wg_long", length=150e-6)
wg_short = siepic.Waveguide(name="wg_short", length=50e-6)


# define a PCell using simphony components
@gf.cell
def mzi():
    c = gf.Component("mzi")

    ysplit = c << ysplitter.component

    gcin = c << gcinput.component

    gcout = c << gcoutput.component

    yrecomb = c << yrecombiner.component

    yrecomb.move(destination=(0, -55.5))
    gcout.move(destination=(-20.4, -55.5))
    gcin.move(destination=(-20.4, 0))

    gcinput["pin1"].connect(ysplitter, gcin, ysplit)
    gcoutput["pin1"].connect(yrecombiner["pin1"], gcout, yrecomb)
    ysplitter["pin2"].connect(wg_long)
    yrecombiner["pin3"].connect(wg_long)
    ysplitter["pin3"].connect(wg_short)
    yrecombiner["pin2"].connect(wg_short)

    wg_long_ref = gf.routing.get_route_from_steps(
        ysplit.ports["pin2"],
        yrecomb.ports["pin3"],
        steps=[{"dx": 91.75 / 2}, {"dy": -61}],
    )
    wg_short_ref = gf.routing.get_route_from_steps(
        ysplit.ports["pin3"],
        yrecomb.ports["pin2"],
        steps=[{"dx": 47.25 / 2}, {"dy": -50}],
    )

    wg_long.path = wg_long_ref
    wg_short.path = wg_short_ref

    c.add(wg_long_ref.references)
    c.add(wg_short_ref.references)

    c.add_port("o1", port=gcin.ports["pin2"])
    c.add_port("o2", port=gcout.ports["pin2"])

    return c


c = mzi()
c.show() # open in KLayout
c.to_3d().show("gl") # 3D visualization


# run the layout aware simulation
with Simulation() as sim:
    l = Laser(name="laser", power=1)
    l.freqsweep(187370000000000.0, 199862000000000.0)
    l.connect(gcinput.pins["pin2"])

    d = Detector(name="detector")
    d.connect(gcoutput.pins["pin2"])

    results = sim.layout_aware_simulation(c)

# Plot the results
f = l.freqs
for run in results:
    p = []
    for sample in run:
        for data_list in sample:
            for data in data_list:
                p.append(data)
    plt.plot(f, p)

run = results[0]
p = []
for sample in run:
    for data_list in sample:
        for data in data_list:
            p.append(data)
plt.plot(f, p, "k")
plt.title("MZI Layout Aware Monte Carlo")
plt.show()
