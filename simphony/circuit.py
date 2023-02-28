"""
Define circuit and connections in simphony.
"""

from typing import List, Set, Tuple, Union
from collections import defaultdict
from itertools import count

from simphony.models import Model, Port, OPort, EPort


class Circuit:
    """
    A circuit tracks connections between ports.

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
    def __init__(self, name: str) -> None:
        self.name = name
        self._components = [] # list of components (model instances) in the circuit
        self._onodes: List[Tuple[OPort, OPort]] = [] # optical connections
        self._enodes: List[Set[EPort]] = [] # electrical connections
        self._next_oidx = count() # netid iterator
        self._next_eidx = count() # netid iterator

    @property
    def components(self) -> List[Model]:
        """
        Return a list of components (model instances) in the circuit.
        
        Order in which models were added is not preserved.

        Returns
        -------
        list
            List of all model instances in the circuit.
        """
        oinstances = [port.instance for pair in self._onodes for port in pair]
        einstances = [wire.instance for node in self._enodes for wire in node]
        return list(set(oinstances + einstances))

    def _connect_o(self, port1: OPort, port2: OPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        vairable on the ports themselves."""
        self._onodes.append((port1, port2))
        port1._connections.add[port2]
        port2._connections.add[port1]

    def _connect_e(self, port1: EPort, port2: EPort) -> None:
        """Connect two ports in the internal netlist and update the connections
        vairable on the ports themselves."""
        def update_connections(enodes: Set[EPort]):
            for eport in enodes:
                eport._connections.update(enodes)
                eport._connections.remove(eport)

        for i, eports in enumerate(self._enodes):
            # EPort already has some connections in the netlist
            if port1 in eports or port2 in eports:
                self._enodes[i].update([port1, port2])
                update_connections(self._enodes[i])
                return
            
        # EPort has not yet appeared in the netlist
        self._enodes.append(set([port1, port2]))
        update_connections(self._enodes[-1])

    def connect(self, port1: Union[Model, Port], port2: Union[Model, Port]):
        """
        Connect two ports together and add to the internal netlist.

        If a Model is passed, the next available optical port is inferred.
        If no optical ports are available, the next available electronic port
        is inferred. The type of ``port1`` is used to determine the type of
        ``port2`` if an explicit port is not given.

        Parameters
        ----------
        port1 : Model or Port
            The first port to be connected.
        port2 : Model or Port
            The second port to connect to.

        Raises
        ------
        ValueError
            If the ports are of the wrong type or are incompatible (i.e. 
            optical to electronic connection).
        """
        def o2x(self, port1: OPort, port2: Union[Model, OPort]):
            """Connect an optical port to a second port (type-inferred)."""
            if isinstance(port2, OPort):
                self._connect_o(port1, port2)
            elif issubclass(type(port2), Model):
                self._connect_o(port1, port2.next_unconnected_oport())
            else:
                raise ValueError(f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})")
            
        def e2x(self, port1: EPort, port2: Union[Model, EPort]):
            """Connect an electronic port to a second port (type-inferred)."""
            if isinstance(port2, EPort):
                self._connect_e(port1, port2)
            elif issubclass(type(port2), Model):
                self._connect_e(port1, port2.next_unconnected_eport())
            else:
                raise ValueError(f"Port types must match or be an instance of Model ({type(port1)} != {type(port2)})")

        if isinstance(port1, OPort):
            o2x(self, port1, port2)

        if isinstance(port1, EPort):
            e2x(self, port1, port2)

        if issubclass(type(port1), Model):
            if p1:=port1.next_unconnected_oport():
                o2x(self, p1, port2)
            elif p1:=port1.next_unconnected_eport():
                e2x(self, p1, port2)
            else:
                raise ValueError(f"Ports must be optical, electronic, or a Model (got '{type(port1)}')")

    def to_model(self) -> Model:
        pass
    