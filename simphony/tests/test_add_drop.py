from matplotlib import pyplot as plt
from phidl import quickplot
from simphony.libraries import siepic
from simphony.simulation import Simulation, Laser, Detector

def ring_factory(radius):
  """Creates a full ring (with terminator) from a half ring.

  Resulting pins are ('pass', 'in', 'out').

  Parameters
  ----------
  radius : float
      The radius of the ring resonator, in meters.
  """
  # Have rings for selecting out frequencies from the data line.
  # See SiPANN's model API for argument order and units.
  halfring1 = siepic.HalfRing(radius=radius)
  halfring2 = siepic.HalfRing(radius=radius)
  terminator = siepic.Terminator()

  halfring1.rename_pins("midb", "pass", "midt", "in")
  halfring2.rename_pins("midt", "out", "midb", "term")

  # the interface method will connect all of the pins with matching names
  # between the two components together
  halfring1.interface(halfring2)
  halfring2["term"].connect(terminator)

  # bundling the circuit as a Subcircuit allows us to interact with it
  # as if it were a component
  return halfring1.circuit.to_subcircuit()

wg_input = siepic.Waveguide(name="wginput", length=100e-6)
quickplot(wg_input.device)
plt.show()
wg_out1 = siepic.Waveguide(name="wgoutput1", length=100e-6)
quickplot(wg_out1.device)
plt.show()
wg_connect1 = siepic.Waveguide(name="wgc1", length=100e-6)
quickplot(wg_connect1.device)
plt.show()
wg_out2 = siepic.Waveguide(name="wg_output2", length=100e-6)
quickplot(wg_out2.device)
plt.show()
wg_connect2 = siepic.Waveguide(name="wg_c2", length=100e-6)
quickplot(wg_connect2.device)
plt.show()
wg_out3 = siepic.Waveguide(name="wg_output3", length=100e-6)
quickplot(wg_out3.device)
plt.show()
terminator = siepic.Terminator(name="term")
quickplot(terminator.device)
plt.show()

ring1 = ring_factory(10e-6)
ring1.name = "ring1"
quickplot(ring1.device)
plt.show()
ring2 = ring_factory(11e-6)
ring2.name = "ring2"
ring3 = ring_factory(12e-6)
ring3.name = "ring3"

ring1.multiconnect(wg_connect1, wg_input["pin2"], wg_out1)
ring2.multiconnect(wg_connect2, wg_connect1, wg_out2)
ring3.multiconnect(terminator, wg_connect2, wg_out3)
print('After connecting rings')
quickplot(ring1.device)
plt.show()
with Simulation() as sim:
    l = Laser()
    l.freqsweep(3e8/1600e-6, 3e8/1500e-6)
    l.connect(wg_input)
    d = Detector().connect(wg_out1)

    data = sim.layout_aware_simulation()
