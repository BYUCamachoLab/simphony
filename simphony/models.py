"""Base class for defining models."""

from __future__ import annotations

import collections.abc
import logging
from copy import deepcopy
from functools import lru_cache, wraps
from itertools import count
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    from simphony.utils import jax

    JAX_AVAILABLE = False
import skrf

from simphony.exceptions import ModelValidationError
from simphony.utils import wl2freq

if TYPE_CHECKING:
    from simphony.circuit import Circuit


log = logging.getLogger(__name__)


_NAME_REGISTER = set()


class Port:
    """Port abstract base class containing name and reference to Model
    instance."""

    def __init__(
        self, name: str, instance: Optional[Union[Model, Circuit]] = None
    ) -> None:
        self.name: str = name
        self.instance: Optional[Model] = instance
        self._connections = set()  # a list of all other ports the port is connected to

    def __repr__(self) -> str:
        """Return a string representation of the port."""
        return f'<{self.__class__.__name__} "{self.name}"{" (connected)" if self.connected else ""} at {hex(id(self))}>'

    def __iter__(self):
        yield self

    def __deepcopy__(self, memo):
        """Deep copy the circuit.

        Copied pins lose connections and reference to an instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.name = deepcopy(self.name, memo)
        result.instance = None
        result._connections = set()
        return result

    def rename(self, name: str) -> None:
        """Rename the port.

        Parameters
        ----------
        name : str
            The new name of the port.
        """
        self.name = name

    @property
    def connected(self) -> bool:
        """If the port is connected to other port(s)."""
        return bool(self._connections)

    def connect_to(self, port: Port) -> None:
        """Connect the port to another port.

        Parameters
        ----------
        port : Port
            The port to connect to.

        Raises
        ------
        NotImplementedError
            Implemented by subclasses.
        """
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
    """Base model class that all components should inherit from.

    Models perform some basic validation on their subclasses, like
    making sure functions required for simulation are present. Some
    functions, such as those that calculate and return scattering
    parameters, are cached to reduce memory usage and calculation times.

    The model tracks its own ports. Ports should not be interacted with
    directly, but should be modified through the functions that Model
    provides.

    Attributes
    ----------
    ocount : int, optional
        The number of optical ports on the model.
    onames : list of str, optional
        The names of the optical ports.
    ecount : int, optional
        The number of electrical ports on the model.
    enames : list of str, optional
        The names of the electrical ports.

    Notes
    -----
    Models are the building blocks of simphony. Without the ability to define
    custom models or use models from other libraries, simphony essentially
    loses all functionality.

    Models must have certain attributes defined. These are:

    - `onames` or `ocount`. ``onames`` is a list of strings, equal in length to
        the number of ports on the device. ``ocount`` is simply the number of
        ports, and port names will be generated automatically as ``"o1"``,
        ``"o2"``, etc. Only declare one or the other; ``onames`` and ``ocount``
        are mutually exclusive.
    - `enames` or `ecount`: the names of the electrical ports or the number of'
        electrical ports, respectively. Rules for `onames` and `ocount` apply
        here as well, except that the auto-generated names will be ``"e1"``,
        ``"e2"``, etc.
    - `s_params`: a function that returns the scattering parameters of the
        model. This function should take a wavelength (in microns) as an
        argument and return a 2D numpy array of the scattering parameters. If
        your calculation is in frequency or SI units, be sure to make the
        appropriate conversions! You can use convenience functions in the
        `simphony.utils` module to help with this.

    Models can optionally have the following attributes defined:

    - `__init__`: the model's constructor, in the case of a parameterized
        model. This function should take any parameters as arguments and
        store them as attributes of the model. Any attributes set here, so long
        as they are saved to the ``self`` attribute, are accessible to the
        `s_params` function.

    An example of a model definition is shown below. This would be equivalent
    to a straight, lossless, broadband waveguide.

    .. code-block:: python

        class MyWaveguide(Model):
            onames = ["in", "out"]

            def __init__(self, length):
                self.length = length

            def s_params(self, wavelength):
                return np.array([[0, 1], [1, 0]])

    Note that the `s_params` function is cached internally by the simulation
    engine. If a model loads s-parameters from a file, for example, this
    significantly cuts down on read times because s-parameters is called on
    each instantiated instance of a model, not just once for each model
    type. Multiple instances of the same model do, however, share the same
    s-parameters if their public attributes (or parameters) are equal. If you
    want to save attributes to the model, but don't want them to be used in the
    caching, prefix the attribute name with an underscore (e.g. `_my_attr`).

    Therefore, the following example is essentially equivalent to what happens
    during a simulation:

    #. Instantiate ``YBranch1``, an instance of YBranch
    #. Instantiate ``YBranch2`` with the same parameters as YBranch1
    #. Call simulation with a given wavelength array
    #. Call s-parameters calculation/loading for YBranch1

       a. Function has not been called previously, cache the result

    #. Call s-parameters calculation/loading for ``YBranch2``

       a. ``YBranch2`` is an instance of YBranch
       b. Check if ``YBranch2 == YBranch1``, ignoring private attributes
       c. Check if arguments (wavelength array) are equal
       d. If equal, return cached result
    """

    # Default keys to ignore when checking for equality or hashing. Private
    # attributes are always ignored when checking for equality.
    _ignore_keys = ["onames", "ocount", "enames", "ecount", "counter"]
    counter = count()

    # These should always be instance attributes, not treated as class
    # attributes. They are set on the instance by the super initializer when
    # rename_ports() is called.
    _oports: list[OPort] = []  # should only be manipulated by rename_oports()
    _eports: list[EPort] = []  # should only be manipulated by rename_eports()

    def __init__(self, name: str = None) -> None:
        if hasattr(self, "_exempt"):
            return

        if hasattr(self, "onames") and hasattr(self, "ocount"):
            raise ModelValidationError(
                "Model defines both 'onames' and 'ocount', which is not allowed."
            )
        if hasattr(self, "onames"):
            self.rename_oports(getattr(self, "onames"))
        elif hasattr(self, "ocount"):
            self.rename_oports([f"o{i}" for i in range(getattr(self, "ocount"))])
        else:
            raise ModelValidationError(
                "Model does not define 'onames' or 'ocount', which is required."
            )

        if hasattr(self, "enames"):
            self.rename_eports(getattr(self, "enames"))
        elif hasattr(self, "ecount"):
            self.rename_eports([f"o{i}" for i in range(getattr(self, "ecount"))])

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

    def __init_subclass__(cls, **kwargs):
        """Ensures subclasses define required functions and automatically calls
        the super().__init__ function."""
        if cls.s_params == Model.s_params:
            raise ModelValidationError(
                f"Model '{cls.__name__}' does not define the required method 's_params(self, wl).'"
            )
        super().__init_subclass__(**kwargs)

    def __eq__(self, other: Model):
        """Compares instance dictionaries to determine equality."""
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        d1 = deepcopy(self.__dict__)
        d2 = deepcopy(other.__dict__)
        # Remove keys that should be ignored
        for key in self._ignore_keys:
            d1.pop(key, None), d2.pop(key, None)
        # Remove private variable keys
        private_attrs = [key for key in d1.keys() if key.startswith("_")]
        for attr in private_attrs:
            d1.pop(attr, None), d2.pop(attr, None)
        try:
            return d1 == d2
        except ValueError as e:
            print(d1)
            print(d2)
            raise e

    def __hash__(self):
        """Hashes the instance dictionary to calculate the hash."""
        try:
            s = frozenset(
                [
                    (k, v)
                    for k, v in vars(self).items()
                    if (
                        not k.startswith("_")
                        and (k not in self._ignore_keys)
                        and isinstance(k, collections.abc.Hashable)
                        and isinstance(v, collections.abc.Hashable)
                    )
                ]
            )
        except TypeError:
            raise ModelValidationError("Model is not hashable.")
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
        for port in result._oports + result._eports:
            port.instance = result
        return result

    def copy(self) -> Model:
        """Deep copy the model instance."""
        return deepcopy(self)

    def __repr__(self) -> str:
        """Code representation of the model."""
        return f'<{self.__class__.__name__} "{self.name}" (o: [{", ".join(["+"+o.name if o.connected else o.name for o in self._oports])}], e: [{", ".join(["+"+e.name if e.connected else e.name for e in self._eports]) or None}])>'

    def __iter__(self):
        """Iterate over unconnected ports."""
        for port in self._oports + self._eports:
            if not port.connected:
                yield port

    @lru_cache
    def _s(self, wl: tuple[float]) -> jnp.ndarray:
        """Internal function to cache the s_params function.

        Parameters
        ----------
        wl : tuple of float
            The wavelength(s) to calculate scattering parameters for. This
            needs to be a hashable type for the caching to work. It is
            internally converted to a numpy array.

        Returns
        -------
        s : ndarray
            Scattering parameters for the model. The shape of the array should
            be (len(wl), n, n) where n is the number of ports.
        """
        # https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls
        log.debug("Cache miss (%s)", str(self))
        return self.s_params(jnp.array(wl))

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the model."""
        self._name = value

    def s_params(self, wl: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """Function to be implemented by subclasses to return scattering
        parameters.

        Parameters
        ----------
        wl : float or ndarray
            The wavelength(s) to calculate scattering parameters for.
            Wavelength is assumed to be in microns.

        Returns
        -------
        s : ndarray
            Scattering parameters for the model. The shape of the array should
            be (len(wl), n, n) where n is the number of ports.
        """
        raise NotImplementedError

    def o(self, value: Union[str, int, None] = None):
        """Get a reference to an optical port.

        Parameters
        ----------
        value : str or int, optional
            The port name or index to get (default None). If not provided, next
            unconnected port is returned.
        """
        if value is not None:
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

    def e(self, value: Union[str, int, None] = None) -> Union[EPort, None]:
        """Get a reference to an electrical port.

        Parameters
        ----------
        value : str or int, optional
            The port name or index to get (default None). If not provided, next
            unconnected port is returned.
        """
        if value is not None:
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

    def rename_oports(self, names: list[str]):
        """Rename all optical ports.

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
            self._oports = list(OPort(name, self) for name in names)
        elif len(names) == len(self._oports):
            _ = [port.rename(name) for port, name in zip(self._oports, names)]
        else:
            raise ModelValidationError(
                f"Number of renamed ports must be equal to number of current ports ({len(names)}!={len(self.onames)})"
            )

    def rename_eports(self, names: list[str]):
        """Rename all electrical ports.

        Parameters
        ----------
        names : list of str
            A list of strings renaming all the ports sequentially. List must
            be as long as existing port count.

        Examples
        --------
        c = ThermalPhaseShifter()
        c.rename_oports(["gnd", "bias"])
        """
        if len(self._eports) == 0:
            self._eports = list(EPort(name, self) for name in names)
        elif len(names) == len(self._eports):
            _ = [port.rename(name) for port, name in zip(self._eports, names)]
        else:
            raise ValueError(
                f"Number of renamed ports must be equal to number of current ports ({len(names)}!={len(self.enames)})"
            )

    def next_unconnected_oport(self) -> Union[OPort, None]:
        """Gets the next unconnected optical port, or None if all connected.

        Returns
        -------
        OPort
            The next unconnected port, or None if all connected.
        """
        for o in self._oports:
            if not o.connected:
                return o

    def next_unconnected_eport(self) -> Union[EPort, None]:
        """Gets the next unconnected electronic port, or None if all connected.

        Returns
        -------
        EPort
            The next unconnected port, or None if all connected.
        """
        for e in self._eports:
            if not e.connected:
                return e

    def is_connected(self) -> bool:
        """Determines if this component is connected to any others.

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

    def to_network(
        self, wl: Union[float, list[float], tuple[float], jnp.ndarray]
    ) -> skrf.Network:
        """Converts the component to a scikit-rf network.

        Parameters
        ----------
        wl : float or ndarray
            The wavelength(s) to calculate scattering parameters for.
            Wavelength is assumed to be in microns.

        Returns
        -------
        Network
            The network representation of the component.
        """
        wl = jnp.asarray(wl).reshape(-1)
        s = self._s(tuple(wl.tolist()))
        f = wl2freq(wl * 1e-6)
        return skrf.Network(f=f[::-1], s=s[::-1], f_unit="Hz")


def clear_name_register():
    """Clears the name register."""
    _NAME_REGISTER.clear()
    Model.counter = count()
