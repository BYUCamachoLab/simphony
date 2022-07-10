from typing import List, TYPE_CHECKING, Tuple
import numpy as np
from phidl.device_layout import Device, DeviceReference
from phidl.geometry import grid
import phidl.routing as pr
from phidl import quickplot
from zmq import device

from simphony.libraries import siepic
from simphony.models import Subcircuit
from simphony.simulation import Detector, Laser, Simulation

if TYPE_CHECKING:
    from simphony import Model
    from simphony.pins import Pin

class Die(Device):
    def __init__(self, name:str = "", *args, **kwargs):

        self.device: Device = Device(name=name)
        self.device_grid: Device = Device(name=f'{name}_grid')
        self.device_list: List[Device] = []
        self.device_refs: List[DeviceReference] = []
        self.connect_refs: List[DeviceReference] = []
        self.grid: List[Device] = []
        self.spacing: Tuple = ()
        super().__init__(name=name, *args, **kwargs)

    def add_components(self, components: List["Model"]):
        for component in components:
            self.device_list.append(component.device)
            self.device_refs.append(self.device.add_ref(component.device))
            component.die = self

    def connect(self, component1: "Model", component2: "Model", pin1: "Pin", pin2: "Pin"):
        if isinstance(component2, siepic.Waveguide):
            if component2.pins.index(pin2) == 0:
                pin_ = pin2._component.pins[1]
            else:
                pin_ = pin2._component.pins[0]
            if None not in (pin1._connection, pin_._connection) and not isinstance(pin2._component, (Subcircuit, Simulation, Laser, Detector)):
                # ==================================
                #  NOTE DO NOT REMOVE THESE LINES!!!
                # ==================================
                # ref = pr.route_smooth(self.device_grid.references[self.device_list.index(component1.device)].ports[pin1.name], self.device_grid.references[self.device_list.index(pin_._connection._component.device)].ports[pin_._connection.name], radius=1, path_type='manhattan', width=pin2._component.width * 1e6, length=pin2._component.length * 1e6)
                # self.device_grid.add(ref)
                
                port1 = self.device_grid.references[self.device_list.index(component1.device)].ports[pin1.name]
                port2 =  self.device_grid.references[self.device_list.index(pin_._connection._component.device)].ports[pin_._connection.name]
                dot = np.dot(port1.normal, port2.normal.T)
                if round((dot[0,0] + dot[1,1]) - (dot[1,0] + dot[0,1])) == 0 and np.intersect1d(port1.normal, port2.normal) is not []:
                    route_path = pr.route_smooth(port1, port2, path_type='L')
                elif round((dot[0,0] + dot[1,1]) - (dot[1,0] + dot[0,1])) == 0 and np.intersect1d(port1.normal, port2.normal) is []:
                    route_path = pr.route_smooth(port1, port2, path_type='J', length1=component2.length * 1e6 / 4, length2=component2.length * 1e6 / 4)
                elif round((dot[0,0] + dot[1,1]) - (dot[1,0] + dot[0,1])) != 0 and np.intersect1d(port1.normal, port2.normal) is not []:
                    route_path = pr.route_smooth(port1, port2, path_type='U', radius=1, width=component2.width * 1e6, length1=component2.length * 1e6 / 3)
                elif round((dot[0,0] + dot[1,1]) - (dot[1,0] + dot[0,1])) != 0 and np.intersect1d(port1.normal, port2.normal) is []:
                    route_path = pr.route_smooth(port1, port2, path_type='C', length1=component2.length * 1e6 / 5, lengtth2=component2.length * 1e6 / 5, left1=component2.length * 1e6 / 5)
                self.device_grid.add(route_path)
        elif not isinstance(pin1._connection._component, (siepic.Waveguide, Subcircuit, Simulation, Laser, Detector)):
            if self.device_grid_refs[self.device_list.index(component2.device)].parent.ports[pin2.name].orientation == 0 or 180:
                overlap = np.linalg.norm(self.device_grid_refs[self.device_list.index(component2.device)].parent.ports[pin2.name].midpoint + self.device_grid_refs[self.device_list.index(component1.device)].parent.ports[pin1.name].midpoint) + self.spacing[0]
            elif self.device_grid_refs[self.device_list.index(component2.device)].parent.ports[pin2.name].orientation == 90 or 270:
                overlap = np.linalg.norm(self.device_grid_refs[self.device_list.index(component2.device)].parent.ports[pin2.name].midpoint + self.device_grid_refs[self.device_list.index(component1.device)].parent.ports[pin1.name].midpoint) + self.spacing[1]
            self.device_grid_refs[self.device_list.index(component1.device)].connect(self.device_grid_refs[self.device_list.index(component1.device)].parent.ports[pin1.name], self.device_grid_refs[self.device_list.index(component2.device)].parent.ports[pin2.name], overlap=overlap)

    def distribute_devices(self, elements='all', direction="x", shape=None, spacing=100, separation=True):
        if direction == 'x':
            self.distribute(elements=elements, direction='x', spacing=spacing, separation=separation)
            self.grid = []
        elif direction == 'y':
            self.distribute(elements=elements, direction='y', spacing=spacing, separation=separation)
            self.grid = []
        elif direction == 'grid':
            self.spacing = spacing
            self.device_grid = grid(device_list=self.device_list, spacing=self.spacing, separation=separation, shape=shape)
            self.device_grid_refs: DeviceReference = self.device_grid.references

    def write_gds(self, filename, unit=0.000001, precision=1e-9, auto_rename=True, max_cellname_length=28, cellname="toplevel"):
        return super().write_gds(filename, unit, precision, auto_rename, max_cellname_length, cellname)

    def write_svg(self, outfile, scaling=10, style=None, fontstyle=None, background="#222", pad="5%", precision=None):
        return super().write_svg(outfile, scaling, style, fontstyle, background, pad, precision)

    def visualize(self):
        quickplot(self.device_grid)
