from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import phidl.routing as pr
from phidl import quickplot, set_quickplot_options
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
        for i, component in enumerate(components):

            # Add component's `device` attribute to a list to
            # keep track of all the devices in the Die
            self.device_list.append(component.device)

            # Add component's `device` attribute to `device_grid` and store the
            # corresponding DeviceReference
            self.device_refs.append(
                self.device_grid.add_ref(component.device, alias=component)
            )

            # Update center position if it is different during instantiation
            self.device_grid.references[i].center = component.device.center

            # Update component's `die` attribute
            component.die = self

    def _connect(
        self, component1: "Model", component2: "Model", pin1: "Pin", pin2: "Pin"
    ):
        """
        Function that connects devices in the Die together when components are hooked up, without the user's knowledge.

        Attributes
        ----------
        component1 :
            One of the connection components
        component2 :
            The other connection component
        pin1 :
            The pin of component1 that component2 has to
            be connected to
        pin2 :
            The pin of component2 that has to be connected
            to component1 (pin1)
        """

        # If component2 is a SiEPIC Waveguide, route the two
        # components it is connecte to together
        if isinstance(component2, siepic.Waveguide):

            # pin_ is the other Waveguide pin that is not pin2
            if component2.pins.index(pin2) == 0:
                pin_ = pin2._component.pins[1]
            else:
                pin_ = pin2._component.pins[0]

            # Route the waveguides together
            self._route_waveguides(component1, component2, pin1, pin_)

            for i, ref in enumerate(self.device_grid.references):
                if ref.parent.name == component2.device.name:
                    self.device_grid.references.pop(i)

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

    def _route_waveguides(self, component1, component2, pin1, pin_) -> None:
        """
        Waveguide routing routine. This function looks at the ports' orientations and
        positions and decides what sort of Waveguide bend is needed, and how to connect
        them.
        """

        # Do nothing if the other pin of Waveguide is not connected to anything or if
        # it is connected to a Simulation, Subcircuit, Laser, Detector
        if None in (pin1._connection, pin_._connection) or isinstance(
            pin_._component, (Subcircuit, Simulation, Laser, Detector)
        ):
            return

        # Port 1 of the connection
        port1 = self.device_grid.references[
            self.device_list.index(component1.device)
        ].ports[pin1.name]

        # Port 2 of the connection
        port2 = self.device_grid.references[
            self.device_list.index(pin_._connection._component.device)
        ].ports[pin_._connection.name]

        # Take dot product of the two ports' normals. Needed to compare their orientations
        dot = np.dot(port1.normal, port2.normal.T)

        # Compare orientations of ports and route them up
        if round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) == 0:
            route_path = self._connect_orthogonal_ports(component2, pin_, port1, port2)

        elif (
            round((dot[0, 0] + dot[1, 1]) - (dot[1, 0] + dot[0, 1])) == -1
            and np.intersect1d(port1.normal, port2.normal) is not []
        ):
            route_path = (
                pr.route_smooth(
                    port1,
                    port2,
                    path_type="straight",
                    radius=1,
                    width=pin_._component.width * 1e6,
                )
                if np.linalg.norm(port1.midpoint - port2.midpoint)
                == component2.length * 1e6
                else pr.route_smooth(
                    port1,
                    port2,
                    path_type="C",
                    length1=component2.length * 1e6 / 5,
                    length2=component2.length * 1e6 / 5,
                    left1=component2.length * 1e6 / 5,
                    radius=1,
                    width=pin_._component.width * 1e6,
                )
            )

        else:
            route_path = self._connect_parallel_ports(component2, pin_, port1, port2)

        route_path.name = f"wg_{pin_._component.device.name}"
        self.device_grid.add_ref(route_path)

        if not isinstance(component2, siepic.Waveguide):
            self.device_list.remove(component2.device)

    def _connect_parallel_ports(self, component2, pin_, port1, port2) -> Device:
        """
        Connect parallel ports. These can either be a 'U' connection or
        a 'C' connection.
        """
        if np.intersect1d(port1.normal, port2.normal) is not []:
            route_path = pr.route_smooth(
                port1,
                port2,
                path_type="U",
                radius=1,
                width=pin_._component.width * 1e6,
                length1=component2.length * 1e6 / 3,
            )
        elif np.intersect1d(port1.normal, port2.normal) is []:
            route_path = pr.route_smooth(
                port1,
                port2,
                path_type="C",
                length1=component2.length * 1e6 / 5,
                length2=component2.length * 1e6 / 5,
                left1=component2.length * 1e6 / 5,
                radius=1,
                width=pin_._component.width * 1e6,
            )

        return route_path

    def _connect_orthogonal_ports(self, component2, pin_, port1, port2) -> Device:
        """
        Connect orthogonal ports. These can be either an 'L' connection,
        or a 'J' connection.
        """
        if np.intersect1d(port1.normal, port2.normal) is not []:
            route_path = pr.route_smooth(
                port1, port2, path_type="L", radius=1, width=pin_._component.width * 1e6
            )
        elif np.intersect1d(port1.normal, port2.normal) is []:
            route_path = pr.route_smooth(
                port1,
                port2,
                path_type="J",
                length1=component2.length * 1e6 / 4,
                length2=component2.length * 1e6 / 4,
                radius=1,
                width=pin_._component.width * 1e6,
            )

        return route_path

    def _interface(self, component1, component2, pin1, pin2):
        for port1 in self.device_grid_refs[self.device_list.index(component1.device)].ports:
            for port2 in self.device_grid_refs[self.device_list.index(component2.device)].ports:
                if port1 == port2:
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

    def _disconnect(self, component1, component2):
        """
        Disconnect the devices in the grid as the pins are disconnected.

        Attributes
        ----------
        component1 :
            One of the connection components
        component2 :
            The other connection component
        """
        if isinstance(component2, siepic.Waveguide):

            # Disconnect components connected by the waveguide
            ref: DeviceReference
            for i, ref in enumerate(self.device_grid.references):
                if ref.parent.name == f"wg_{component2.name}":
                    self.device_grid.references.pop(i)

        elif not isinstance(
            component2,
            (siepic.Waveguide, Subcircuit, Simulation, Laser, Detector),
        ):
            self.device_grid_refs[self.device_list.index(component1.device)].center = [
                0,
                0,
            ]

            self.device_grid_refs[self.device_list.index(component2.device)].center = [
                0,
                0,
            ]

    def distribute_devices(
        self, direction="x", shape=None, spacing=10, separation=True
    ):
        """
        Distribute the components in the Die i.e. space them apart. They can be arranged
        either along the x-axis, or the y-axis, or in a grid.

        Attributes
        ----------
        direction :
            The direction to arrange the components along.
            Can be 'x', 'y', or 'grid'
        shape :
            Shape of the grid, if `direction` is 'grid'
        spacing :
            The spacing between the components. Takes a single integer/float
            value if direction is 'x' or 'y', and a tuple of integer/float values
            if direction is 'grid'
        """
        if direction == "x":

            # Set the spacing attribute
            self.spacing = (spacing, spacing)

            # Distribute along x
            device_grid: Device = self.device_grid.distribute(
                elements="all", direction="x", spacing=spacing, separation=separation
            )

            # Update alias names
            names = [name for name, _ in self.device_grid.aliases.items()]
            device_grid.aliases = dict(zip(names, list(device_grid.aliases.values())))

            # Update `device_grid` and `device_grid_refs` attributes
            self.device_grid = device_grid
            self.device_grid_refs: DeviceReference = self.device_grid.references

        elif direction == "y":

            # Set the spacing attribute
            self.spacing = (spacing, spacing)

            # Distribute along y
            device_grid: Device = self.device_grid.distribute(
                elements="all", direction="y", spacing=spacing, separation=separation
            )

            # Update alias names
            names = [name for name, _ in self.device_grid.aliases.items()]
            device_grid.aliases = dict(zip(names, list(device_grid.aliases.values())))

            # Update `device_grid` and `device_grid_refs` attributes
            self.device_grid = device_grid
            self.device_grid_refs: DeviceReference = self.device_grid.references

        elif direction == "grid":

            # Set the spacing attribute
            self.spacing = spacing

            # Distribute along a grid
            device_grid: Device = grid(
                device_list=self.device_list,
                spacing=self.spacing,
                separation=separation,
                shape=shape,
            )

            # Update alias names
            names = [name for name, _ in self.device_grid.aliases.items()]
            device_grid.aliases = dict(zip(names, list(device_grid.aliases.values())))

            # Update `device_grid` and `device_grid_refs` attributes
            self.device_grid = device_grid
            self.device_grid_refs: DeviceReference = self.device_grid.references

    def visualize(
        self, show_ports=True, show_subports=False, label_aliases=True, font_size=8
    ):
        """
        Visualize the layout. This opens a new matplotlib window with the layout drawing.

        Attributes
        ----------
        show_ports :
            Whether to show the ports on the layout.
        show_subports :
            Whether to show the subports on the layout.
        label_aliases :
            Whether to label the components on the layout.
        font_size :
            The font size of the labels.
        """

        # Set plot parameters
        set_quickplot_options(
            show_ports=show_ports,
            show_subports=show_subports,
            label_aliases=label_aliases,
        )

        # Plot the layout
        quickplot(self.device_grid.references)
