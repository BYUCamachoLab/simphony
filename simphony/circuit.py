"""Define circuit and connections in simphony."""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import copy, deepcopy
from itertools import count
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

from simphony.connect import connect_s, vector_innerconnect_s
from simphony.exceptions import ModelValidationError
from simphony.models import EPort, Model, OPort, Port

if TYPE_CHECKING:
    from simphony.simulation.simdevices import SimDevice

log = logging.getLogger(__name__)


class Circuit(Model):
    """A circuit tracks connections between ports.

    Examples
    --------
    .. code-block:: python

        gc_in = GratingCoupler()
        y_split = YJunction()
        wg_short = Waveguide()
        wg_long = Waveguide()
        y_combine = YJunction()
        gc_out = GratingCoupler

        cir = Circuit()
        cir.connect(gc_in.o(1), y_split)
        cir.connect(y_split, wg_short)
        cir.connect(y_split, wg_long)
        cir.connect(wg_short, y_combine.o(1))
        cir.connect(wg_long, y_combine.o(2))
        cir.connect(y_combine, gc_out.o(1))

        cir.connect(gc_in.o(0), Laser())
        cir.connect(gc_out.o(0), Detector())
    """

    def __init__(self, name: str = None) -> None:
        self.name = name or "circuit"

        self._components = []  # list of model instances in the circuit
        self._onodes: list[tuple[OPort, OPort]] = []  # optical netlist
        self._enodes: list[set[EPort]] = []  # electrical netlist

        self._sparams = None
        self.exposed = False
        self.sim_devices: list[SimDevice] = []

    # def __copy__(self):
    #     """Shallow copy the circuit."""
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     result.__dict__.update(self.__dict__)
    #     return result

    # def __deepcopy__(self, memo):
    #     """Deep copy the circuit."""
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         setattr(result, k, deepcopy(v, memo))
    #     return result

    @property
    def components(self) -> list[Model | Circuit]:
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
        """Return a list of all unconnected optical ports in the circuit."""
        return [
            port
            for comp in self._components
            for port in comp._oports
            if not port.connected
        ]

    @property
    def _eports(self):
        """Return a list of all unconnected electrical ports in the circuit."""
        return [
            port
            for comp in self._components
            for port in comp._eports
            if not port.connected
        ]

    def port_info(self) -> str:
        """Return a string containing information about the ports in the
        circuit.

        Returns
        -------
        str
            String containing information about the ports in the circuit.
        """
        info = f"{self.name} port info:\n"
        for comp in self._components:
            info += f"{comp}:\n"
            for port in comp._oports:
                info += f"  {'|' if port.connected else 'O'} {port}\n"
        return info

    def _connect_o(self, port1: OPort, port2: OPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        variable on the ports themselves."""

        if not any(port1.instance is comp for comp in self._components):
            self._components.append(port1.instance)
        if not any(port2.instance is comp for comp in self._components):
            self._components.append(port2.instance)

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

    def connect(
        self,
        port1: Model | OPort | EPort,
        port2: Model | OPort | EPort | list[Model | OPort | EPort],
    ) -> None:
        """Connect two ports together and add to the internal netlist.

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
            Connections will be made by iterating through unconnected ports
            in their declared order.
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

        def o2x(self, port1: OPort, port2: Model | OPort):
            """Connect an optical port to a second port (type-inferred)."""
            if isinstance(port2, OPort):
                self._connect_o(port1, port2)
            elif issubclass(type(port2), Model) or isinstance(port2, Circuit):
                self._connect_o(port1, port2.next_unconnected_oport())
            else:
                raise ValueError(
                    f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})"
                )

        def e2x(self, port1: EPort, port2: Model | EPort):
            """Connect an electronic port to a second port (type-inferred)."""
            if isinstance(port2, EPort):
                self._connect_e(port1, port2)
            elif issubclass(type(port2), Model) or isinstance(port2, Circuit):
                self._connect_e(port1, port2.next_unconnected_eport())
            else:
                raise ValueError(
                    f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})"
                )

        # Single optical port to second port
        if isinstance(port1, OPort):
            return o2x(self, port1, port2)

        # Single electronic port to second port
        if isinstance(port1, EPort):
            return e2x(self, port1, port2)

        # If the first port is a model, we're doing a multiconnect.
        # Iterate through the second object until we run out of ports.
        if issubclass(type(port1), Model):
            for i, p2 in enumerate(list(port2)):
                # If the model has unconnected optical ports, use those first.
                if p1 := port1.next_unconnected_oport():
                    o2x(self, p1, p2)
                # Otherwise, try to use the electronic ports.
                elif p1 := port1.next_unconnected_eport():
                    e2x(self, p1, p2)
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

    def __iter__(self):
        yield self

    def s_params(self, wl):
        """Compute the scattering parameters for the circuit.

        Using the sub-network growth algorithm.
        """
        # TODO: What if a different wavelength range is passed in? This seems
        # like a bad idea.
        if self._sparams is not None:
            return self._sparams

        if len(self._onodes) == 0:
            raise ModelValidationError("No connections in circuit.")

        class STemp:
            def __init__(self, s, oports):
                self._sparams = s
                self._oports = oports

            def s_params(self, wl):
                return self._sparams

        wl = tuple(wl)

        # Iterate through all connections and update the s-parameters
        for port1, port2 in self._onodes:
            log.debug(
                "Connecting %s from %s to %s from %s",
                port1,
                port1.instance,
                port2,
                port2.instance,
            )
            # innerconnect
            if port1.instance == port2.instance:
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

            model._oports = [
                port
                for port in model._oports
                if port is not port1 and port is not port2
            ]
            for port in model._oports:
                port.instance = model

        # Make sure all remaining ports point to the same model, otherwise we
        # have a disconnected circuit.
        for port in self._oports:
            if port.instance is not model:
                raise ValueError(
                    "Separated circuit: two or more disconnected subcircuits contained within the same circuit."
                )

        self._sparams = model._sparams
        return self._sparams
