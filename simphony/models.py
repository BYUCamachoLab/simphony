"""
Base class for defining models.
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Union
from functools import lru_cache, wraps

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False

from simphony.exceptions import ModelValidationError


class Port:
    """
    Port base class containing name and reference to Model instance.
    """

    def __init__(self, name: str, instance: Model = None) -> None:
        self.name = name
        self.instance = instance
        self._connections = set()  # a list of all other ports the port is connected to

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} "{self.name}" at {hex(id(self))}>'

    def rename(self, name: str) -> None:
        self.name = name

    @property
    def connected(self) -> bool:
        """If the port is connected to other port(s)."""
        return bool(self._connections)

    def connect_to(self, port: Port) -> None:
        raise NotImplementedError


class OPort(Port):
    """Optical ports can only be connected to one other port."""

    def connect_to(self, port: OPort) -> None:
        if not isinstance(port, OPort):
            raise ValueError(
                f"Optical ports can only be connected to other optical ports (got '{type(port).__name__}')"
            )
        elif not self._connections:
            self._connections.add(port)
        else:
            raise ConnectionError(f"Port '{self.name}' is already connected!")


class EPort(Port):
    """Electrical ports can be connected to many other ports."""

    def connect_to(self, port: EPort) -> None:
        if not isinstance(port, EPort):
            raise ValueError(
                f"Electrical ports can only be connected to other electrical ports (got '{type(port).__name__}')"
            )
        else:
            self._connections.add(port)


class Model:
    """
    Base model class that all components should inherit from.

    Models perform some basic validation on their subclasses, like making sure
    functions required for simulation are present. Some functions, such as
    those that calculate and return scattering parameters, are cached to reduce
    memory usage and calculation times.

    The model tracks its own ports. Ports should not be interacted with
    directly, but should be modified through the functions that Model provides.
    """

    _oports: List[OPort] = []  # should only be manipulated by rename_oports()
    _eports: List[EPort] = []  # should only be manipulated by rename_eports()
    _ignore_keys = [
        "onames",
        "ocount",
        "enames",
        "ecount",
    ]  # ignore when checking for equality or hashing

    def __init__(self) -> None:
        if hasattr(self, "onames"):
            self.rename_oports(self.onames)
        if hasattr(self, "ocount"):
            self.rename_oports([f"o{i}" for i in range(self.ocount)])
        if not hasattr(self, "_oports"):
            raise ModelValidationError(
                "Model does not define 'onames' or 'ocount', which is required."
            )

    def __init_subclass__(cls) -> None:
        """
        Ensures subclasses define required functions and automatically calls
        the super().__init__ function.
        """
        if not hasattr(cls, "s_params"):
            raise ModelValidationError(
                f"Model '{cls.__name__}' does not define the required method 's_params(self, wl).'"
            )

        orig_init = cls.__init__

        @wraps(orig_init)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            super(self.__class__, self).__init__()

        cls.__init__ = __init__

    def __eq__(self, other: Model):
        """Compares instance dictionaries to determine equality."""
        d1 = deepcopy(self.__dict__)
        d2 = deepcopy(other.__dict__)
        for key in self._ignore_keys:
            d1.pop(key), d2.pop(key)
        return d1 == d2

    def __hash__(self):
        """Hashes the instance dictionary to calculate the hash."""
        s = frozenset(
            [
                (k, v)
                for k, v in vars(self).items()
                if (not k.startswith("_") and (k not in self._ignore_keys))
            ]
        )
        return hash(s)

    def __copy__(self):
        """Shallow copy the circuit."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """Deep copy the circuit."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __repr__(self) -> str:
        """Code representation of the circuit."""
        return f'<{self.__class__.__name__} at {hex(id(self))} (o: [{", ".join(["+"+o.name if o.connected else o.name for o in self._oports])}], e: [{", ".join(["+"+e.name if e.connected else e.name for e in self._eports]) or None}])>'

    @lru_cache
    def _s(self, wl):
        # https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls
        return self.s_params(wl)

    def o(self, value: Union[str, int] = None):
        """
        Get a reference to an optical port.

        Parameter
        ---------
        value : str or int, optional
            The port name or index to get (default None). If not provided, next
            unconnected port is returned.
        """
        if value:
            if isinstance(value, str):
                for p in self._oports:
                    if p.name == value:
                        return p
            elif isinstance(value, int):
                return self._oports[value]
            else:
                raise ValueError("Port indexer must be a name (str) or index (int).")
        else:
            return self.next_unconnected_oport()

    def e(self, value: Union[str, int] = None):
        """
        Get a reference to an electrical port.

        Parameter
        ---------
        value : str or int, optional
            The port name or index to get (default None). If not provided, next
            unconnected port is returned.
        """
        if value:
            if isinstance(value, str):
                for p in self._eports:
                    if p.name == value:
                        return p
            elif isinstance(value, int):
                return self._eports[value]
            else:
                raise ValueError("Port indexer must be a name (str) or index (int).")
        else:
            return self.next_unconnected_eport()

    def rename_oports(self, names: List[str]) -> Model:
        """
        Rename all optical ports.

        Parameters
        ----------
        names : list of str
            A list of strings renaming all the ports sequentially. List must
            be as long as existing port count.

        Examples
        --------
        c = Coupler()
        c.rename_oports(["in", "through", "add", "drop"])
        """
        if len(self._oports) == 0:
            self._oports = set(Port(name, self) for name in names)
        elif len(names) == len(self.onames):
            (port.rename(name) for port, name in zip(self._oports, names))
        else:
            raise ValueError(
                f"Number of renamed ports must be equal to number of current ports ({len(names)}!={len(self.onames)})"
            )

    def next_unconnected_oport(self) -> Optional[OPort]:
        """
        Gets the next unconnected optical port, or None if all connected.

        Returns
        -------
        OPort
            The next unconnected port, or None if all connected.
        """
        for o in self._oports:
            if not o.connected:
                return o

    def next_unconnected_eport(self) -> Optional[EPort]:
        """
        Gets the next unconnected electronic port, or None if all connected.

        Returns
        -------
        EPort
            The next unconnected port, or None if all connected.
        """
        for e in self._eports:
            if not e.connected:
                return e

    def is_connected(self) -> bool:
        """
        Determines if this component is connected to any others.

        Returns
        -------
        bool
            True if any optical or electronic ports are connected to others.
        """
        if any([o.connected for o in self._oports]):
            return True
        if any([e.connected for e in self._eports]):
            return True
        return False


if __name__ == "__main__":

    class Coupler(Model):
        onames = ["o1", "o2", "o3", "o4"]

        def __init__(self, k):
            self.k = k

        def s_params(self, wl):
            print(f"cache miss ({self.k})")
            return wl * 1j * self.k

    class Waveguide(Model):
        ocount = 2
        ecount = 2

        def __init__(self, a):
            self.a = a

        def s_params(self, wl):
            print(f"cache miss ({self.a})")
            return 1j * self.a

    class Heater(Model):
        jit = False

        def __init__(self, onames=["o0", "o1"]):
            self.onames = onames

        def s_params(self, wl):
            pass

    # m = Model()
    c1 = Coupler(0.5)
    c2 = Coupler(0.6)
    c1.onames = ["p1", "p2", "p3", "p4"]

    print(c1.onames)
    print(c2.onames)

    pass
