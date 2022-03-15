import sys
from simphony.layout import Circuit

sys.path.append("C:\\Users\\12269\\Documents\\GitHub\\simphony\\simphony")

from simphony.libraries import siepic
from models import Model
from simphony.simulators import Simulator

wg1 = siepic.Waveguide()
wg2 = siepic.Waveguide()
Model.pin_count = 1
model1 = Model()
model2 = Model()

print(f'model1.__hash__() before connection = {model1.__hash__()}\n')
print(f'model2.__hash__() before connection = {model2.__hash__()}\n')

model1.connect(wg1)
model2.connect(wg2)

print(f'model1.__hash__() after connection = {model1.__hash__()}\n')
print(f'model2.__hash__() after connection = {model2.__hash__()}\n')
