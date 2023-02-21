"""
Base class for defining models.
"""

from __future__ import annotations
from copy import deepcopy

import warnings
from types import SimpleNamespace
from typing import List, Union
from functools import lru_cache, wraps

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    def jit(func, *args, **kwargs):
        """Mock "jit" version of a function. Warning is only raised once."""
        warnings.warn("Jax not available, cannot compile using 'jit'!")
        return func

    jax = SimpleNamespace(jit=jit)
    JAX_AVAILABLE = False

from simphony.exceptions import ModelValidationError


class Port:
    """Port base class containing name and reference to Model instance."""
    def __init__(self, name, instance=None):
        self.name = name
        self.instance = instance
        self.connections = [] # a list of all other ports the port is connected to

    def __repr__(self):
        return f'<{self.__class__.__name__} "{self.name}" at {hex(id(self))}>'

    def rename(self, name):
        self.name = name


class OPort(Port):
    """Optical ports can only be connected to one other port."""
    pass


class EPort(Port):
    """Electrical ports can be connected to many other ports."""
    pass


class Model:
    """
    Base model class that all components should inherit from.

    Models perform some basic validation on their subclasses, like making sure
    functions required for simulation are present. Some functions, such as 
    those that calculate and return scattering parameters, are cached to reduce
    memory usage and calculation times.

    The model tracks its own ports. Ports should not be interacted with 
    directly, but should be modified through the functions that Model provides.

    The Model base class will attempt to wrap your function with JAX's "jit"
    functionality, which will speed up the calculation of the s-parameters when
    running on a GPU or TPU. If you want to disable jit, set ``jit=False`` as
    a class variable when you write the class.
    """
    _oports: List[OPort] = [] # should only be manipulated by rename_oports()
    _eports: List[EPort] = [] # should only be manipulated by rename_eports()
    _ignore_keys = ["onames", "ocount", "enames", "ecount", "jit"] # ignore when checking for equality or hashing
    jit = True
    
    def __init__(self) -> None:
        if hasattr(self, "onames"):
            self.rename_oports(self.onames)
        if hasattr(self, "ocount"):
            self.rename_oports([f"o{i}" for i in range(self.ocount)])
        if not hasattr(self, "_oports"):
            raise ModelValidationError("Model does not define 'onames' or 'ocount', which is required.")

    def __init_subclass__(cls) -> None:
        """
        Ensures subclasses define required functions and automatically calls
        the super().__init__ function.
        """
        if not hasattr(cls, "s_params"):
            raise ModelValidationError(f"Model '{cls.__name__}' does not define the required method 's_params(self, wl).'")
        if cls.jit:
            cls.s_params = jax.jit(cls.s_params)
        
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
        s = frozenset([(k, v) for k, v in vars(self).items() if (not k.startswith("_") and (k not in self._ignore_keys))])
        return hash(s)

    @lru_cache
    def _s(self, wl):
        # https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls
        return self.s_params(wl)
    
    def o(self, value):
        """Get a reference to an optical port."""
        if isinstance(value, str):
            for p in self._oports:
                if p.name == value:
                    return p
        elif isinstance(value, int):
            return self._oports[value]
        else:
            raise ValueError("Port indexer must be a name (str) or index (int).")
        
    def e(self, value):
        """Get a reference to an electrical port."""
        if isinstance(value, str):
            for p in self._eports:
                if p.name == value:
                    return p
        elif isinstance(value, int):
            return self._eports[value]
        else:
            raise ValueError("Port indexer must be a name (str) or index (int).")
    
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
            self._oports = [Port(name, self) for name in names]
        elif len(names) == len(self.onames):
            (port.rename(name) for port, name in zip(self._oports, names))
        else:
            raise ValueError(f"Number of renamed ports must be equal to number of current ports ({len(names)}!={len(self.onames)})")
        
    def next_unconnected_oport(self) -> OPort:
        """
        Gets the next unconnected optical port, or None if all connected.

        Returns
        -------
        OPort
            The next unconnected port.
        """
        for o in self._oports:
            if not o.connections:
                return o
            
    def next_unconnected_eport(self) -> EPort:
        """
        Gets the next unconnected electronic port, or None if all connected.

        Returns
        -------
        EPort
            The next unconnected port.
        """
        for e in self._eports:
            if not e.connections:
                return e


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
