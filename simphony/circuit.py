"""Define circuit and connections in simphony."""

from __future__ import annotations

from collections import defaultdict
from itertools import count
from typing import List, Optional, Set, Tuple, Union

from simphony.connect import connect_s, vector_innerconnect_s
from simphony.exceptions import ModelValidationError
from simphony.models import EPort, Model, OPort, Port


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

        # self._components = []  # list of model instances in the circuit

        self._internal_oports: list[OPort] = []  # internal optical ports
        self._internal_eports: list[EPort] = []  # internal electrical ports
        self._oports: list[OPort] = []  # exposed optical ports
        self._eports: list[EPort] = []  # exposed electrical ports

        self._onodes: list[tuple[OPort, OPort]] = []  # optical connections
        self._enodes: list[set[EPort]] = []  # electrical connections

        self._next_oidx = count()  # netid iterator
        self._next_eidx = count()  # netid iterator

        self._sparams = None
        self.exposed = False
        self.sim_devices: list[SimDevice] = []

    @property
    def components(self) -> list[Model | Circuit]:
        """Return a list of components (model instances) in the circuit.

        Order in which models were added is not preserved.

        Returns
        -------
        list
            List of all model instances in the circuit.
        """
        oinstances = [port._original_instance for pair in self._onodes for port in pair]
        einstances = [wire._original_instance for node in self._enodes for wire in node]
        return list(set(oinstances + einstances))
        # return self._components

    def _update_ports(self, model1: Model | Circuit, model2: Model | Circuit) -> None:
        """Update the internal list of ports in the circuit."""
        if model1 not in self.components:
            self._internal_oports.extend(model1._oports)
            self._internal_eports.extend(model1._eports)
        if model2 not in self.components:
            self._internal_oports.extend(model2._oports)
            self._internal_eports.extend(model2._eports)

    def _connect_o(self, port1: OPort, port2: OPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        vairable on the ports themselves."""
        self._update_ports(port1.instance, port2.instance)
        self._onodes.append((port1, port2))
        port1._connections.add(port2)
        port2._connections.add(port1)

    def _connect_e(self, port1: EPort, port2: EPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        vairable on the ports themselves."""

        def update_connections(enodes: set[EPort]):
            for eport in enodes:
                eport._connections.update(enodes)
                eport._connections.remove(eport)

        self._update_ports(port1.instance, port2.instance)
        for i, eports in enumerate(self._enodes):
            # EPort already has some connections in the netlist
            if port1 in eports or port2 in eports:
                self._enodes[i].update([port1, port2])
                update_connections(self._enodes[i])
                return

        # EPort has not yet appeared in the netlist
        self._enodes.append({port1, port2})
        update_connections(self._enodes[-1])

    def connect(self, port1: Model | Port, port2: Model | Port):
        """Connect two ports together and add to the internal netlist.

        If a Model is passed, the next available optical port is inferred. If
        no optical ports are available, the next available electronic port is
        inferred. The type of ``port1`` is used to determine the type of
        ``port2`` if an explicit port is not given.

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

    def expose(
        self,
        ports: Port | list[Port] | None = None,
        names: str | list[str] | None = None,
    ) -> None:
        """Expose ports to the outside world.

        Parameters
        ----------
        ports : Port or list of Ports
            The ports to expose. If not given, all unconnected ports are
            exposed.
        names : str or list of str
            The names to assign to the ports. If not given, the names are
            inferred from the port names.
        """
        new_ports = []
        if ports is None:
            ports = [port for port in self._internal_oports if not port.connected]
            ports += [port for port in self._internal_eports if not port.connected]

        if isinstance(ports, Port):
            ports = [ports]
        if names is None:
            names = [port.name for port in ports]
        if isinstance(names, str):
            names = [names]
        if len(ports) != len(names):
            raise ValueError(
                f"Number of ports ({len(ports)}) does not match number of names ({len(names)})"
            )

        new_ports = ports  # don't make new ports

        new_oports = [port for port in new_ports if isinstance(port, OPort)]
        new_eports = [port for port in new_ports if isinstance(port, EPort)]
        self._oports = new_oports
        self._eports = new_eports
        self.exposed = True

    def __iter__(self):
        yield self

    def s_params(self, wl):
        """Compute the scattering parameters for the circuit.

        Using the sub-network growth method.
        """
        # TODO: What if a different wavelength range is passed in? This seems
        # like a bad idea.
        if self._sparams is not None:
            return self._sparams

        if len(self._onodes) == 0:
            raise ModelValidationError("No devices in circuit.")
        if not self.exposed:
            self.expose()

        class STemp:
            def __init__(self, s, oports):
                self.s = s
                self._oports = oports

            def s_params(self, wl):
                return self.s

        # Iterate through all connections and update the s-parameters
        for cnx in self._onodes:
            port1, port2 = cnx
            if port1.instance == port2.instance:
                sparams = port1.instance.s_params(wl)
                p1_idx = port1.instance._oports.index(port1)
                p2_idx = port2.instance._oports.index(port2)
                model = STemp(
                    vector_innerconnect_s(sparams, p1_idx, p2_idx),
                    port1.instance._oports,
                )
            else:
                p1_sparams = port1.instance.s_params(wl)
                p1_idx = port1.instance._oports.index(port1)
                p2_sparams = port2.instance.s_params(wl)
                p2_idx = port2.instance._oports.index(port2)
                model = STemp(
                    connect_s(p1_sparams, p1_idx, p2_sparams, p2_idx),
                    port1.instance._oports + port2.instance._oports,
                )
            model._oports.remove(port1)
            model._oports.remove(port2)
            port1._update_instance(model)
            port2._update_instance(model)

        self._sparams = model.s
        return self._sparams
