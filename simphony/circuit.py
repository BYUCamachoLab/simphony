"""Define circuit and connections in simphony."""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import copy, deepcopy
from itertools import count
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from simphony.connect import connect_s, vector_innerconnect_s
from simphony.exceptions import ModelValidationError, SeparatedCircuitError
from simphony.models import _NAME_REGISTER, EPort, Model, OPort, Port

log = logging.getLogger(__name__)


class Circuit(Model):
    """A circuit is a netlist that tracks connections between ports.

    Attributes
    ----------
    name : str
        Name of the circuit.
    components : list
        List of all model instances in the circuit. Models are added to the
        circuit by connecting ports of individual instances.

    Parameters
    ----------
    name : str, optional
        Name of the circuit.

    Examples
    --------
    We'll create an MZI circuit:

    .. code-block:: python

        gc_in = GratingCoupler()
        y_split = YJunction()
        wg_short = Waveguide(length=100)
        wg_long = Waveguide(length=150)
        y_combine = YJunction()
        gc_out = GratingCoupler

        cir = Circuit()

        # You can explicitly connect ports:
        cir.connect(gc_in, y_split)
        # Or, you can automatically zip ports together. "connect" will
        # repeatedly get the next unconnected port until either the left runs
        # out of ports or the right list is exhausted.
        cir.connect(y_split, [wg_short, wg_long])
        cir.connect(y_combine, gc_out)
        cir.connect(y_combine, [wg_short, wg_long])

        cir.connect(gc_in, Laser())
        cir.connect(gc_out, Detector())

    You can also instantiate from a list of connections:

    .. code-block:: python

        cir = Circuit().from_connections(
            [
                (gc_in, y_split),
                (y_split, [wg_short, wg_long]),
                (y_combine, gc_out),
                (y_combine, [wg_short, wg_long]),
                (gc_in, Laser()),
                (gc_out, Detector()),
            ]
        )
    """

    _exempt = True

    def __init__(self, name: str = None) -> None:
        if name:
            if name in _NAME_REGISTER:
                raise ValueError(
                    f"Name '{name}' is already in use. Please choose a different name."
                )
            else:
                _NAME_REGISTER.add(name)
                self._name = name
        else:
            name = self.__class__.__name__ + str(next(self.counter))
            while name in _NAME_REGISTER:
                name = self.__class__.__name__ + str(next(self.counter))
            else:
                _NAME_REGISTER.add(name)
                self._name = name

        self._components = []  # list of model instances in the circuit
        self._onodes: list[tuple[OPort, OPort]] = []  # optical netlist
        self._enodes: list[set[EPort]] = []  # electrical netlist

        # Exposed ports point from a ports belonging to the
        # original component port to a new port pointing to the circuit.
        self.exposed_ports = {}

    def __iter__(self):
        yield self

    def __deepcopy__(self, memo):
        """Deep copy the circuit."""
        MANUAL_COPY = ["_components", "_onodes", "_enodes", "exposed_ports"]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in MANUAL_COPY:
                setattr(result, k, deepcopy(v, memo))

        setattr(result, "_components", [])
        setattr(result, "_onodes", [])
        setattr(result, "_enodes", [])
        setattr(result, "exposed_ports", {})

        # Copying strategy:
        # Copy all components, which each get their own port instances.
        # Iterate through the connections in onodes and enodes, find their
        # respective instances in the copied components, and connect them.
        # The copied components are not actually set on the resulting circuit,
        # but get added to the circuit when they are connected.
        components_copy = [deepcopy(comp, memo) for comp in self.components]

        def get_component_index(port):
            """Given a port, gets the index of its parent component in the
            components list."""
            for idx, comp in enumerate(self._components):
                if comp is port.instance:
                    return idx

        def get_oport_index(port, component):
            """Given a port and its parent component, gets the index of the
            port in the component's oport list."""
            for idx, p in enumerate(component._oports):
                if p.name == port.name:
                    return idx

        def get_eport_index(port, component):
            """Given a port and its parent component, gets the index of the
            port in the component's eport list."""
            for idx, p in enumerate(component._eports):
                if p.name == port.name:
                    return idx

        for o1, o2 in self._onodes:
            # Get the index of the oports' parents in the original component list
            o1_component_idx = get_component_index(o1)
            o2_component_idx = get_component_index(o2)
            # Get the corresponding oports in the copied component list
            o1_port_idx = get_oport_index(o1, components_copy[o1_component_idx])
            o2_port_idx = get_oport_index(o2, components_copy[o2_component_idx])

            result._connect_o(
                components_copy[o1_component_idx].o(o1_port_idx),
                components_copy[o2_component_idx].o(o2_port_idx),
            )

        # It is essential that the order of the exposed ports in the copy is
        # the same as the order of the exposed ports in the original circuit.
        for oport in self.exposed_ports.keys():
            cidx = get_component_index(oport)
            pidx = get_oport_index(oport, components_copy[cidx])

            # Create a new port on the copied circuit and point it to the
            # appropriate copied port from the copied component.
            new_port = deepcopy(oport)
            new_port.instance = result
            result.exposed_ports[components_copy[cidx].o(pidx)] = new_port

        # for oport in self._internal_oports:
        #     cidx = get_component_index(oport)
        #     pidx = get_oport_index(oport, components_copy[cidx])

        #     # If the port is one of the exposed ports,
        #     if oport in self.exposed_ports.keys():
        #         # Create a new port on the copied circuit and point it to the
        #         # appropriate copied port from the copied component.
        #         new_port = deepcopy(oport)
        #         new_port.instance = result
        #         result.exposed_ports[components_copy[cidx].o(pidx)] = new_port

        for es in self._enodes:
            indices = []
            ports = []
            for e_n in es:
                # Get the index of the eports' parents in the original component list
                component_idx = get_component_index(e_n)
                indices.append(component_idx)

                # Get the corresponding eports in the copied component list
                port_idx = get_eport_index(e_n, components_copy[component_idx])
                ports.append(port_idx)

            for idx1, idx2, p1, p2 in zip(indices, indices[1:], ports, ports[1:]):
                result._connect_e(
                    components_copy[idx1].e(p1), components_copy[idx2].e(p2)
                )

            # TODO: Implement exposed eports
            # # For each of the two ports,
            # for oi, cidx, pidx in zip([o1, o2], [o1_component_idx, o2_component_idx], [o1_port_idx, o2_port_idx]):
            #     # If the port is one of the exposed ports,
            #     if oi in self.exposed_ports.values():
            #         # Create a new port on the copied circuit and point it to the
            #         # appropriate copied port from the copied component.
            #         new_port = deepcopy(oi)
            #         new_port.instance = result
            #         result.exposed_ports[new_port] = components_copy[cidx].o(pidx)

        return result

    def copy(self) -> Circuit:
        """Return a deep copy of the circuit.

        Note that while a circuit preserves insertion order, the insertion
        order is not preserved in the copy.

        Returns
        -------
        Circuit
            Deep copy of the circuit.
        """
        return deepcopy(self)

    def __eq__(self, other: Circuit):
        if self.components != other.components:
            return False
        if self._onodes != other._onodes:
            return False
        if self._enodes != other._enodes:
            return False
        return super().__eq__(other)

    def __hash__(self):
        """Hashes the instance dictionary to calculate the hash."""
        return super().__hash__()

    @property
    def components(self) -> List[Union[Model, Circuit]]:
        """Return a list of components (model instances) in the circuit in the
        order they were added.

        Returns
        -------
        list
            List of all model instances in the circuit.
        """
        return self._components

    @property
    def _oports(self):
        """Return a list of all exposed ports, or if none are exposed, all
        unconnected optical ports in the circuit."""
        if self.exposed_ports:
            return [
                port for port in self.exposed_ports.values() if isinstance(port, OPort)
            ]
        return [
            port
            for comp in self.components
            for port in comp._oports
            if not port.connected
        ]

    @property
    def _internal_oports(self):
        """Return a list of all unconnected optical ports in the circuit."""
        return [
            port
            for comp in self.components
            for port in comp._oports
            if not port.connected
        ]

    @property
    def _eports(self):
        """Return a list of all unconnected electrical ports in the circuit."""
        if self.exposed_ports:
            return [
                port for port in self.exposed_ports.values() if isinstance(port, EPort)
            ]
        return [
            port
            for comp in self.components
            for port in comp._eports
            if not port.connected
        ]

    # TODO: Add net ID/number to the output so it's easier to see what's
    # connected to what.
    def port_info(self) -> str:
        """Return a string containing information about the ports in the
        circuit.

        Returns
        -------
        str
            String containing information about the ports in the circuit.
        """
        info = ""
        comps = self.components
        for comp in comps:
            info += f"{comp}:\n"
            for port in comp._oports:
                info += f"  {'|' if port.connected else 'O'} {port}\n"
        header = f'"{self.name}" contains {len(comps)} models:\n'
        return header + info

    def plot_networkx(self):
        """Plot the circuit optical port connections using NetworkX.

        Returns
        -------
        fig, ax
            Figure and axis objects.
        """

        G = nx.Graph()

        for o1, o2 in self._onodes:
            G.add_edge(
                f"{o1.instance.name}",
                f"{o2.instance.name}",
                label=f"{o1.name} : {o2.name}",
            )

        for i, (oldport, newport) in enumerate(self.exposed_ports.items()):
            G.add_edge(f"{oldport.instance.name}", f"o[{i}]")

        options = {
            # "font_size": 12,
            # "node_size": 3000,
            # "node_color": "white",
            # "node_shape": "s",
            "node_color": None,
            "edgecolors": "black",
            # "linewidths": 5,
            # "width": 5,
            "bbox": dict(
                facecolor="skyblue", edgecolor="black", boxstyle="round,pad=0.2"
            ),
        }
        edge_options = {
            # "font_size": 12,
            # "font_weight": "bold",
        }

        pos = nx.spring_layout(G, k=0.35)
        nx.draw_networkx(G, pos, with_labels=True, **options)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=nx.get_edge_attributes(G, "label"), **edge_options
        )

        plt.tight_layout()
        return plt.gcf(), plt.gca()

    def _connect_o(self, port1: OPort, port2: OPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        variable on the ports themselves."""

        if not any(port1.instance is comp for comp in self._components):
            self._components.append(port1.instance)
        if not any(port2.instance is comp for comp in self._components):
            self._components.append(port2.instance)

        # Don't double-connnect optical ports
        existing_connections = [
            port for connection in self._onodes for port in connection
        ]
        if port1 in existing_connections:
            raise ValueError(f"Port '{port1.name}' is already connected.")
        if port2 in existing_connections:
            raise ValueError(f"Port '{port2.name}' is already connected.")

        self._onodes.append((port1, port2))
        port1._connections.add(port2)
        port2._connections.add(port1)

    def _connect_e(self, port1: EPort, port2: EPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        variable on the ports themselves."""

        def update_connections(enodes: set[EPort]):
            for eport in enodes:
                eport._connections.update(enodes)  # add all other ports
                eport._connections.remove(eport)  # remove self

        if not any(port1.instance is comp for comp in self._components):
            self._components.append(port1.instance)
        if not any(port2.instance is comp for comp in self._components):
            self._components.append(port2.instance)

        # Check if EPort already has some connections in the netlist
        for i, eports in enumerate(self._enodes):
            if port1 in eports or port2 in eports:
                self._enodes[i].update([port1, port2])
                update_connections(self._enodes[i])
                return

        # EPort has not yet appeared in the netlist
        self._enodes.append({port1, port2})
        update_connections(self._enodes[-1])

    def _o2x(self, port1: OPort, port2: Union[Model, OPort]):
        """Connect an optical port to a second port (type-inferred)."""
        if isinstance(port2, OPort):
            self._connect_o(port1, port2)
        elif issubclass(type(port2), Model) or isinstance(port2, Circuit):
            self._connect_o(port1, port2.next_unconnected_oport())
        else:
            raise ValueError(
                f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})"
            )

    def _e2x(self, port1: EPort, port2: Union[Model, EPort]):
        """Connect an electronic port to a second port (type-inferred)."""
        if isinstance(port2, EPort):
            self._connect_e(port1, port2)
        elif issubclass(type(port2), Model) or isinstance(port2, Circuit):
            self._connect_e(port1, port2.next_unconnected_eport())
        else:
            raise ValueError(
                f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})"
            )

    def add(self, component: Model) -> Model:
        """Add a component to the circuit.

        Parameters
        ----------
        component : Model or Circuit
            Component to add to the circuit.

        Returns
        -------
        Model
            The component that was added.

        Examples
        --------
        You can add components to a circuit
        >>> cir.add(gc_in)

        See Also
        --------
        connect : Connect components together.
        """
        if component in self.components:
            raise ValueError(f"Component '{component.name}' is already in the circuit.")
        self.components.append(component)
        return component

    def connect(
        self,
        port1: Union[Model, OPort, EPort],
        port2: Union[Model, OPort, EPort, List[Union[Model, OPort, EPort]]],
    ) -> None:
        """Connect two ports together and add to the internal netlist.

        Connections can be defined in the following ways:

        * **port to port** (connect two explicit ports)
        * **port to model** (connect an explicit port to the next available
          port on a model)
        * **model to port** (connect the next available port on a model to an
          explicit port)
        * **model to model** (iterate over the next available ports on a model
          to the next available ports on another model)
        * **model to list of ports** (sequentially connect the next available
          port on a model to a list of explicit ports)
        * **model to list of models** (sequentially connect the next available
          port on a model to the first available port from each model in a
          list)

        If a ``Model`` is passed, the next available optical port is used. If
        no optical ports are available, the next available electronic port is
        used. In this way, the first argument can be thought about as an
        "iterable". Thought about this way, this function acts similarly to the
        ``zip`` function: connections are made between items of the first
        argument and items of the second until one of the two is exhausted, at
        which point remaining items in the longer list are ignored.

        If ``port1`` is explicitly an optical or electronic port and the type
        of ``port2`` is not given, the type of ``port1`` is used to infer the
        type of ``port2``.

        Parameters
        ----------
        port1 : Circuit, Model or Port
            The first port to be connected, or a model with unconnected ports.
            Connections will be made by iterating through unconnected ports in
            their declared order.
        port2 : Circuit, Model, Port, or list
            The second port to connect to, or a list of ports or models.

        Raises
        ------
        ValueError
            If the ports are of the wrong type or are incompatible (i.e.
            optical to electronic connection).

        Examples
        --------
        You can connect two ports, models, or a model to a list of ports.

        We can connect two ports. The type of port1 is used to determine the
        type of port2. And, since port2 is a Model, the next available port is
        inferred.

        >>> cir.connect(gc_in.o(1), y_split)

        We can connect a model to a port. The next available port of the model
        is used to connect to each consecutive port in the list. OPorts are
        resolved first, then EPorts.

        >>> cir.connect(y_split, [wg_short, wg_long])
        """
        # Single optical port to second port
        if isinstance(port1, OPort):
            return self._o2x(port1, port2)

        # Single electronic port to second port
        if isinstance(port1, EPort):
            return self._e2x(port1, port2)

        # If the first port is a model, we're doing a multiconnect.
        # Iterate through the second object until we run out of ports.
        if issubclass(type(port1), Model):
            for i, p2 in enumerate(list(port2)):
                # If the model has unconnected optical ports, use those first.
                if p1 := port1.next_unconnected_oport():
                    self._o2x(p1, p2)
                # Otherwise, try to use the electronic ports.
                elif p1 := port1.next_unconnected_eport():
                    self._e2x(p1, p2)
                # If we run out of ports, raise an error.
                else:
                    if i == 0:
                        raise ValueError(
                            f"Model argument in connect() must have at least one unconnected port."
                        )
            return

        raise ValueError(
            f"Ports must be optical (OPort), electronic (EPort), or a Model/Circuit (got '{type(port1)}' and '{type(port2)}')"
        )

    def from_connections(
        self, connections: List[Tuple[Union[OPort, EPort], Union[OPort, EPort]]]
    ) -> None:
        """Connect a list of ports together.

        Parameters
        ----------
        connections : list[tuple[OPort or EPort, OPort or EPort]]
            A list of tuples of ports to connect together.

        Examples
        --------
        >>> cir.from_connections([(gc_in.o(1), y_split), (y_split, wg_short), (y_split, wg_long)])
        """
        for port1, port2 in connections:
            self.connect(port1, port2)

    def expose(
        self, ports: Union[Union[OPort, EPort], List[Union[OPort, EPort]]]
    ) -> None:
        """Expose a list of ports.

        Scattering parameters of the subcircuit will be computed and indexed
        using the exposed ports.

        Parameters
        ----------
        ports : OPort or EPort or list of OPort or EPort
            Port or list of ports to expose.

        Examples
        --------
        >>> cir.expose([gc_in.o(1), gc_out.o(1)])
        """
        for port in list(ports):
            if port in self.exposed_ports.values():
                raise ValueError(f"Port '{port.name}' is already exposed.")
            if port not in self._internal_oports and port not in self._eports:
                raise ValueError(f"Port '{port.name}' is not in the circuit.")
            new_port = deepcopy(port)
            new_port.instance = self
            self.exposed_ports[port] = new_port

    def s_params(self, wl: Union[float, np.ndarray]) -> np.ndarray:
        """Compute the scattering parameters for the circuit.

        Parameters
        ----------
        wl : float or np.ndarray
            Wavelength at which to compute the scattering parameters.

        Notes
        -----
        Uses the subnetwork growth algorithm [1]_.

        .. [1] R. C. Compton and D. B. Rutledge, "Perspectives in microwave
            circuit analysis," Proceedings of the 32nd Midwest Symposium on
            Circuits and Systems, Champaign, IL, USA, 1989, pp. 716-718 vol.2,
            doi: 10.1109/MWSCAS.1989.101955. url: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=101955&isnumber=3167
        """
        if len(self._onodes) == 0:
            raise ModelValidationError("No connections in circuit.")

        # Automatically expose all unconnected ports if not explicitly exposed
        if not self.exposed_ports:
            self.expose(self._oports)

        class STemp:
            def __init__(self, s, oports):
                self._sparams = s
                self._oports = oports

            def s_params(self, wl):
                return self._sparams

        wl = tuple(np.asarray(wl).reshape(-1))

        ckt_temp = deepcopy(self)

        # Iterate through all connections and update the s-parameters
        for port1, port2 in ckt_temp._onodes:
            log.debug(
                "Connecting %s from %s to %s from %s",
                port1,
                port1.instance,
                port2,
                port2.instance,
            )
            # innerconnect
            if port1.instance is port2.instance:
                log.debug("Innerconnecting %s and %s", port1, port2)
                sparams = (
                    port1.instance._s(wl)
                    if hasattr(port1.instance, "_s")
                    else port1.instance.s_params(wl)
                )
                p1_idx = port1.instance._oports.index(port1)
                p2_idx = port2.instance._oports.index(port2)
                model = STemp(
                    vector_innerconnect_s(sparams, p1_idx, p2_idx),
                    port1.instance._oports,
                )
            # connect
            else:
                log.debug("Connecting %s and %s", port1, port2)
                p1_sparams = (
                    port1.instance._s(wl)
                    if hasattr(port1.instance, "_s")
                    else port1.instance.s_params(wl)
                )
                p1_idx = port1.instance._oports.index(port1)
                p2_sparams = (
                    port2.instance._s(wl)
                    if hasattr(port2.instance, "_s")
                    else port2.instance.s_params(wl)
                )
                p2_idx = port2.instance._oports.index(port2)
                model = STemp(
                    connect_s(p1_sparams, p1_idx, p2_sparams, p2_idx),
                    port1.instance._oports + port2.instance._oports,
                )

            # Remove the ports that were connected from the list of ports
            model._oports = [
                port
                for port in model._oports
                if port is not port1 and port is not port2
            ]
            # Update ports on connected models to point to the new temporary
            # model
            for port in model._oports:
                port.instance = model

        # Make sure all leftover ports in the circuit point to the same model,
        # otherwise we have a disconnected circuit.
        for port in ckt_temp._internal_oports:
            if port.instance is not model:
                raise SeparatedCircuitError(
                    f"Two or more disconnected subcircuits contained within the same circuit."
                )
            # port.instance = ckt_temp

        idx = [model._oports.index(port) for port in ckt_temp.exposed_ports.keys()]
        s = model._sparams[:, idx, :][:, :, idx]
        return s
