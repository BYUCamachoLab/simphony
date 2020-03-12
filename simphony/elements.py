# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import uuid

from scipy.interpolate import interp1d

# FIXME: Is interpolating in frequency better than in wavelength?

class PinList:
    def __init__(self, *args):
        for item in args:
            print(item)
        self.pins = [*args]

    def __getitem__(self, item):
        if type(item) is str:
            idx = self.pins.index(item)
            return idx

    def __setitem__(self, key, value):
        if type(key) is str:
            idx = self.pins.index(key)
            self.pins[idx] = value
        elif type(key) is int:
            self.pins[key] = value
        else:
            raise TypeError

    def __repr__(self):
        return str(self.pins)

    def __len__(self):
        return len(self.pins)
        

class Element:
    """
    The basic element type describing the model for a component with scattering
    parameters.

    Any class that inherits from Element or its subclasses must declare the
    attributes of an element, see Attributes. Following the general EAFP coding
    style of Python, errors will only be raised when an unimplemented function
    is called, not when the class instance is created.

    Attributes
    ----------
    nodes : tuple of str
        A tuple of all the node names for the element.
    wl_bounds : tuple of float
        A tuple of the valid wavelength bounds for the element in the order
        (lower, upper).
    """
    name = None
    # Change to pins? Instead of nodes?
    nodes = None
    wl_bounds = (None, None)

    _ignored_ = ['name', 'nodes', 'wl_bounds']

    def __init__(self, name: str = None):
        self._rename(name)

    def __getattr__(self, name):
        print(name)
        return super().__getattribute__(name)

    def __eq__(self, other: 'Element'):
        # TODO: What if the two instances have different class variable values?
        # As in, it was instantiated with the default but one of them was later
        # changed. These are variables that subclasses implement but that are
        # not required by the parent Element class.
        # parent_attr = dir(Element)
        # child_attr = dir(self)
        if type(self) is type(other):
            not_ignored = [attr for attr in dir(self) if attr not in dir(Element)]
            sdict = {k : getattr(self, k) for k in not_ignored}
            odict = {k : getattr(other, k) for k in not_ignored}
            return sdict == odict
        else:
            return False

    def __ne__(self, other: 'Element'):
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def _node_idx_by_name(self, name: str) -> int:
        """
        Given a string representing a node name, returns which index
        that node represents.

        Since scattering parameters are stored as a matrix, the order of the
        node names corresponds to the order of those ports in the s-matrix.

        Parameters
        ----------
        name : str
            The name of the node.

        Returns
        -------
        idx : int
            The index corresponding to the given node name.
        
        Raises
        ------
        ValueError
            If the node name doesn't exist in the element.
        """
        try:
            return self.nodes.index(name)
        except ValueError:
            raise ValueError('name "{}" not in defined nodes.'.format(name))

    def _rename(self, name: str = None) -> None:
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__ + "_" + str(uuid.uuid4())[:8]


    def _monte_carlo_(self, *args, **kwargs):
        raise NotImplementedError

    def rename_nodes(self, nodes) -> None:
        """
        Renames the nodes for the instance object. Order is preserved and only
        names are remapped.

        Parameters
        ----------
        nodes : tuple
            The string names of the new nodes, in order, as a tuple.
        """
        if type(nodes) is not tuple:
            raise TypeError('nodes must be a tuple, but is {}.'.format(type(nodes)))

        if len(self.nodes) == len(nodes):
            self.nodes = nodes
        else:
            raise ValueError('number of node names provided does not match (needs {})'.format(len(self.nodes)))

    def s_parameters(self, start: float, end: float, num: int):
        """
        Returns scattering parameters for the element with its given 
        parameters.

        Parameters
        ----------
        start : float
            The lower wavelength bound for the simulation.
        end : float
            The upper wavelength bound for the simulation.
        num : int
            The number of points to interpolate between `start` and `end`.

        Returns
        -------
        f, s : float, array
            The frequency range and corresponding scattering parameters.
        
        Raises
        ------
        NotImplementedError
            Raised if the subclassing element doesn't implement this function.
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(resampled, sampled, s_parameters):
        """Returns the result of a cubic interpolation for a given frequency range.

        Parameters
        ----------
        output_freq : np.array
            The desired frequency range for a given input to be interpolated to.
        input_freq : np.array
            A frequency array, indexed matching the given s_parameters.
        s_parameters : np.array
            S-parameters for each frequency given in input_freq.

        Returns
        -------
        result : np.array
            The values of the interpolated function (fitted to the input 
            s-parameters) evaluated at the `output_freq` frequencies.
        """
        func = interp1d(sampled, s_parameters, kind='cubic', axis=0)
        return func(resampled)

# class PElement(Element):
#     """
#     Parameterized Elements have scattering parameters that are calculated on
#     the fly, usually based on instance attributes.
#     """
#     pass

# class SElement(Element):
#     """
#     Static Elements have pre-simulated scattering parameters, often loaded 
#     from a file and independent of instance attributes.
#     """
#     pass
