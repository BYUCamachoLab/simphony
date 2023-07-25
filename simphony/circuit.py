"""Define circuit and connections in simphony."""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import copy, deepcopy
from itertools import count
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

import numpy as np

from simphony.connect import connect_s, vector_innerconnect_s
from simphony.exceptions import ModelValidationError, SeparatedCircuitError
from simphony.models import EPort, Model, OPort, Port

if TYPE_CHECKING:
    from simphony.simulation.simdevices import SimDevice

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
    _next_idx = count()

    def __init__(self, name: str = None) -> None:
        self.name = name or f"circuit{next(self._next_idx)}"

        self._components = []  # list of model instances in the circuit
        self._onodes: list[tuple[OPort, OPort]] = []  # optical netlist
        self._enodes: list[set[EPort]] = []  # electrical netlist

        # These are set when the circuit is simulated and the ports need to
        # correspond to the correct indices in the S matrix.
        self._cascaded_oports: list[OPort] = []
        self._cascaded_eports: list[EPort] = []

        # TODO: Do we want to implement a "frozen" such that, once s_params has
        # been called, connections can no longer be made, changed, etc, since
        # running s_params modifies the underlying ports?

    def __iter__(self):
        yield self

    def __deepcopy__(self, memo):
        """Deep copy the circuit."""
        MANUAL_COPY = ["_components", "_onodes", "_enodes"]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in MANUAL_COPY:
                setattr(result, k, deepcopy(v, memo))

        setattr(result, "_components", [])
        setattr(result, "_onodes", [])
        setattr(result, "_enodes", [])

        # Copying strategy:
        # Copy all components, which each get their own port instances.
        # Iterate through the connections in onodes and enodes, find their
        # respective instances in the copied components, and connect them.
        # The copied components are not actually set on the resulting circuit,
        # but get added to the circuit when they are connected.
        components_copy = [deepcopy(comp, memo) for comp in self.components]

        for o1, o2 in self._onodes:
            # Get the index of the oports' parents in the original component list
            idx1 = next(
                idx for idx, comp in enumerate(self._components) if comp is o1.instance
            )
            idx2 = next(
                idx for idx, comp in enumerate(self._components) if comp is o2.instance
            )
            # Get the corresponding oports in the copied component list
            o1_copy = next(
                idx
                for idx, port in enumerate(components_copy[idx1]._oports)
                if port.name == o1.name
            )
            o2_copy = next(
                idx
                for idx, port in enumerate(components_copy[idx2]._oports)
                if port.name == o2.name
            )
            result._connect_o(
                components_copy[idx1].o(o1_copy), components_copy[idx2].o(o2_copy)
            )

        for e1, e2 in self._enodes:
            # Get the index of the eports' parents in the original component list
            idx1 = next(
                idx for idx, comp in enumerate(self._components) if comp is e1.instance
            )
            idx2 = next(
                idx for idx, comp in enumerate(self._components) if comp is e2.instance
            )
            # Get the corresponding eports in the copied component list
            e1_copy = next(
                idx
                for idx, port in enumerate(components_copy[idx1]._eports)
                if port.name == e1.name
            )
            e2_copy = next(
                idx
                for idx, port in enumerate(components_copy[idx2]._eports)
                if port.name == e2.name
            )
            result._connect_e(
                components_copy[idx1].e(e1_copy), components_copy[idx2].e(e2_copy)
            )

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
    def sim_devices(self) -> list[SimDevice]:
        """Return a list of simulation devices in the circuit in the order they
        were added.

        Returns
        -------
        list
            List of all simulation devices in the circuit.
        """
        return [comp for comp in self.components if isinstance(comp, SimDevice)]

    @property
    def _oports(self):
        """Return a list of all unconnected optical ports in the circuit."""
        if self._cascaded_oports:
            return self._cascaded_oports
        return [
            port
            for comp in self.components
            for port in comp._oports
            if not port.connected
        ]

    @property
    def _eports(self):
        """Return a list of all unconnected electrical ports in the circuit."""
        if self._cascaded_eports:
            return self._cascaded_eports
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

        class STemp:
            def __init__(self, s, oports):
                self._sparams = s
                self._oports = oports

            def s_params(self, wl):
                return self._sparams

        wl = tuple(np.asarray(wl).reshape(-1))

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
        for port in self._oports:
            if port.instance is not model:
                print(port)
                raise SeparatedCircuitError(
                    f"Two or more disconnected subcircuits contained within the same circuit."
                )
            port.instance = self

        self._cascaded_oports = model._oports

        return model._sparams
