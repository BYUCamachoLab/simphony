from random import choice
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import phidl.routing as pr
from phidl import quickplot
from phidl.device_layout import Device, DeviceReference
from phidl.geometry import grid

from simphony.libraries import siepic
from simphony.models import Subcircuit
from simphony.simulation import Detector, Laser, Simulation

if TYPE_CHECKING:
    from simphony import Model
    from simphony.pins import Pin


class Die(Device):
    """
    A Die object. It is built on top of PHIDL's Device class, and can hold the `device`
    attributes of components. It can also automatically route the devices when components
    are connected to each other.

    Attributes
    ----------
    name :
        Name of the die (optional)
    """

    def __init__(self, name: str = "", *args, **kwargs):

        self.device: Device = Device(name=name)  # Holds every device in the die
        self.device_grid: Device = Device(
            name=f"{name}_grid"
        )  # Holds every device in the die if arranged in a grid
        self.device_list: List[Device] = []  # List of devices in the die
        self.device_refs: List[
            DeviceReference
        ] = []  # Holds references to every device in self.device
        self.spacing: Tuple = ()  # Spacing in between components
        super().__init__(name=name, *args, **kwargs)

    def add_components(self, components: List["Model"]):
        """
        Adds components to the die.

        Attributes
        ----------
        components :
            List of components of type `Model` to add.
        """
        for component in components:
            self.device_list.append(component.device)
            self.device_refs.append(self.device_grid.add_ref(component.device))
            component.die = self

    def _connect(
        self, component1: "Model", component2: "Model", pin1: "Pin", pin2: "Pin"
    ):

        if isinstance(component2, siepic.Waveguide):
            if component2.pins.index(pin2) == 0:
                pin_ = pin2._component.pins[1]
            else:
                pin_ = pin2._component.pins[0]
            if None not in (pin1._connection, pin_._connection) and not isinstance(
                pin2._component, (Subcircuit, Simulation, Laser, Detector)
            ):
                # ==================================
                #  NOTE DO NOT REMOVE THESE LINES!!!
                # ==================================
                # ref = pr.route_smooth(self.device_grid.references[self.device_list.index(component1.device)].ports[pin1.name], self.device_grid.references[self.device_list.index(pin_._connection._component.device)].ports[pin_._connection.name], radius=1, path_type='manhattan', width=pin2._component.width * 1e6, length=pin2._component.length * 1e6)
                # self.device_grid.add(ref)

                port1 = self.device_grid.references[
                    self.device_list.index(component1.device)
                ].ports[pin1.name]
                port2 = self.device_grid.references[
                    self.device_list.index(pin_._connection._component.device)
                ].ports[pin_._connection.name]
                dot = np.dot(port1.normal, port2.normal.T)
                if (
                    round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) == 0
                    and np.intersect1d(port1.normal, port2.normal) is not []
                ):
                    route_path = pr.route_smooth(port1, port2, path_type="L", radius=1, width=pin2._component.width * 1e6)
                elif (
                    round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) == 0
                    and np.intersect1d(port1.normal, port2.normal) is []
                ):
                    route_path = pr.route_smooth(
                        port1,
                        port2,
                        path_type="J",
                        length1=component2.length * 1e6 / 4,
                        length2=component2.length * 1e6 / 4,
                        radius=1,
                        width=pin2._component.width * 1e6,
                    )
                elif (
                    round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) == -1
                    and np.intersect1d(port1.normal, port2.normal) is not []
                ):
                    if np.linalg.norm(port1.midpoint - port2.midpoint) == component2.length * 1e6:
                        route_path = pr.route_smooth(
                            port1,
                            port2,
                            path_type="straight",
                            radius=1,
                            width=pin2._component.width * 1e6,
                        )
                    else:
                        route_path = pr.route_smooth(
                            port1,
                            port2,
                            path_type="C",
                            length1=component2.length * 1e6 / 5,
                            length2=component2.length * 1e6 / 5,
                            left1=choice([-1, 1]) * component2.length * 1e6 / 5,
                            radius=1,
                            width=pin2._component.width * 1e6,
                        )
                elif (
                    round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) != 0
                    and np.intersect1d(port1.normal, port2.normal) is not []
                ):
                    route_path = pr.route_smooth(
                        port1,
                        port2,
                        path_type="U",
                        radius=1,
                        width=component2.width * 1e6,
                        length1=component2.length * 1e6 / 3,
                    )
                elif (
                    round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) != 0
                    and np.intersect1d(port1.normal, port2.normal) is []
                ):
                    route_path = pr.route_smooth(
                        port1,
                        port2,
                        path_type="C",
                        length1=component2.length * 1e6 / 5,
                        length2=component2.length * 1e6 / 5,
                        left1=component2.length * 1e6 / 5,
                        radius=1,
                        width=pin2._component.width * 1e6,
                    )
                self.device_grid.add(route_path)
        elif not isinstance(
            pin1._connection._component,
            (siepic.Waveguide, Subcircuit, Simulation, Laser, Detector),
        ):
            if (
                self.device_grid_refs[self.device_list.index(component2.device)]
                .parent.ports[pin2.name]
                .orientation
                == 0
                or 180
            ):
                overlap = (
                    np.linalg.norm(
                        self.device_grid_refs[self.device_list.index(component2.device)]
                        .parent.ports[pin2.name]
                        .midpoint
                        + self.device_grid_refs[
                            self.device_list.index(component1.device)
                        ]
                        .parent.ports[pin1.name]
                        .midpoint
                    )
                    + self.spacing[0]
                )
            elif (
                self.device_grid_refs[self.device_list.index(component2.device)]
                .parent.ports[pin2.name]
                .orientation
                == 90
                or 270
            ):
                overlap = (
                    np.linalg.norm(
                        self.device_grid_refs[self.device_list.index(component2.device)]
                        .parent.ports[pin2.name]
                        .midpoint
                        + self.device_grid_refs[
                            self.device_list.index(component1.device)
                        ]
                        .parent.ports[pin1.name]
                        .midpoint
                    )
                    + self.spacing[1]
                )
            self.device_grid_refs[self.device_list.index(component1.device)].connect(
                self.device_grid_refs[
                    self.device_list.index(component1.device)
                ].parent.ports[pin1.name],
                self.device_grid_refs[
                    self.device_list.index(component2.device)
                ].parent.ports[pin2.name],
                overlap=overlap,
            )

    def distribute_devices(
        self, elements="all", direction="x", shape=None, spacing=100, separation=True
    ):
        if direction == "x":
            self.device_grid.distribute(
                elements=elements, direction="x", spacing=spacing, separation=separation
            )
            self.device_grid_refs: DeviceReference = self.device_grid.references
        elif direction == "y":
            self.device_grid.distribute(
                elements=elements, direction="y", spacing=spacing, separation=separation
            )
            self.device_grid_refs: DeviceReference = self.device_grid.references
        elif direction == "grid":
            self.spacing = spacing
            self.device_grid = grid(
                device_list=self.device_list,
                spacing=self.spacing,
                separation=separation,
                shape=shape,
            )
            self.device_grid_refs: DeviceReference = self.device_grid.references

    def visualize(self):
        if self.device_grid_refs is not []:
            quickplot(self.device_grid)
