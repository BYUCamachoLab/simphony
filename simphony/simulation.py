# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.simulation
===================

This package contains the base classes for running simulations.
"""

import copy
import logging

from typing import List, Optional, Tuple, Union
import numpy as np

from simphony.connect import connect_s, innerconnect_s
from simphony.elements import Model
from simphony.netlist import Element, ElementList, PinList, Subcircuit
from simphony.tools import freq2wl, wl2freq
from numpy import ndarray

_module_logger = logging.getLogger(__name__)


class ScatteringMatrix:
    def __init__(self, freq=None, s=None, pinlist=None):
        self.freq = freq
        self.s = s
        self._pinlist = None
        self.pinlist = pinlist

    @property
    def pinlist(self):
        return self._pinlist

    @pinlist.setter
    def pinlist(self, pinlist):
        if pinlist:
            pinlist.element = self
            self._pinlist = pinlist


class SimulationResult:
    """A simulated block of a circuit; can represent either elements or entire
    subcircuits.

    It is used by Simulation in order to store s-parameters of recursively
    included subcircuits and elements while cascading all blocks into one final
    component representing the circuit as a whole.

    Parameters
    ----------
    component : Component, optional
        A component to initialize the data members of the object.

    Attributes
    ----------
    pins : simphony.netlist.PinList
        An ordered tuple of the external pin names of the simulated component.
    """

    _logger = _module_logger.getChild("SimulationResult")

    def __init__(self, pinlist=None):
        self._pinlist = None
        self.pinlist = pinlist

    @property
    def pinlist(self):
        return self._pinlist

    @pinlist.setter
    def pinlist(self, pinlist):
        self._logger.debug("pinlist property set")
        if pinlist:
            pinlist.element = self
            self._pinlist = pinlist
            assert self.pinlist.element == self

    @property
    def pins(self):
        return tuple([pin.name for pin in self.pinlist])


class SweepSimulationResult(SimulationResult):
    """A simulation result for a swept simulation.

    Parameters
    ----------
    freq : np.array
        A numpy array of the frequency values in its simulation.
    smat : ScatteringMatrix
        A numpy array of the s-parameter matrix for the given frequency range.
    """

    def __init__(self, freq, smat):
        super().__init__(smat.pinlist)
        self.f = freq
        self.s = smat.s

    def data(
        self,
        inp: Union[str, List[int], List[str], int],
        outp: Union[str, int],
        invals: Optional[Union[int, List[int]]] = None,
        dB: bool = False,
    ) -> Tuple[ndarray, ndarray]:
        """
        Parameters
        ----------
        inp : str, Pin, list
            Input pin. Can also be a list of input pins in the case of multi-port input
        outp : str or Pin
            Output pin.
        invals : float, list
            Input values to apply to inp, can be complex. If None, applies 1 each. Defaults to None.
        db : bool
            Whether to return results in dB. Defaults to False.

        Returns
        --------
        output : Requested output of circuit
        """
        # single port input
        if not (
            isinstance(inp, list)
            or isinstance(inp, np.ndarray)
            or isinstance(inp, tuple)
        ):
            inp = [inp]

        # make values if necessary
        if invals is None:
            invals = np.ones_like(inp)
        elif isinstance(invals, int):
            invals = np.array([invals])
        elif isinstance(invals, list) or isinstance(invals, tuple):
            invals = np.array(invals)

        # make sure things line up
        if len(inp) != len(invals):
            raise ValueError("Given different # of inputs and magnitudes")

        # put all inputs together in large array
        idx = [self.pinlist[i].index for i in inp]
        inputs = np.zeros(len(self.pinlist), dtype="complex64")
        inputs[idx] = invals

        # do conversion
        s = np.abs(self.s[:, self.pinlist[outp].index, :] @ inputs) ** 2

        # apply decibels
        if dB:
            s = np.log10(s)

        return self.f, s


class MonteCarloSimulationResult(SimulationResult):
    """
    Parameters
    ----------
    freq : np.ndarray
    smat : simphony.simulation.ScatteringMatrix
    runs : int
    """

    def __init__(self, freq, smat, runs):
        super().__init__(smat.pinlist)
        self.f = freq
        self.results = [smat]
        self.runs = runs

    def add_result(self, result):
        self.results.append(result)

    def data(
        self,
        inp: Union[str, List[int], List[str], int],
        outp: Union[str, int],
        run: int,
        invals: Optional[Union[int, List[int]]] = None,
        dB: bool = False,
    ) -> Tuple[ndarray, ndarray]:
        """
        Parameters
        ----------
        inp : str, Pin, list
            Input pin. Can also be a list of input pins in the case of multi-port input
        outp : str or Pin
            Output pin.
        run : int
            Number of monte-carlo run to return
        invals : float, list
            Input values to apply to inp, can be complex. If None, applies 1 each. Defaults to None.
        db : bool
            Whether to return results in dB. Defaults to False.

        Returns
        --------
        output : Requested output of circuit
        """
        # single port input
        if not (
            isinstance(inp, list)
            or isinstance(inp, np.ndarray)
            or isinstance(inp, tuple)
        ):
            inp = [inp]

        # make values if necessary
        if invals is None:
            invals = np.ones_like(inp)
        elif isinstance(invals, int):
            invals = np.array([invals])
        elif isinstance(invals, list) or isinstance(invals, tuple):
            invals = np.array(invals)

        # make sure things line up
        if len(inp) != len(invals):
            raise ValueError("Given different # of inputs and magnitudes")

        # put all inputs together in large array
        idx = [self.pinlist[i].index for i in inp]
        inputs = np.zeros(len(self.pinlist), dtype="complex64")
        inputs[idx] = invals

        # do conversion
        res = self.results[run]
        s = np.abs(res.s[:, self.pinlist[outp].index, :] @ inputs) ** 2

        # apply decibels
        if dB:
            s = np.log10(s)

        return self.f, s


class Simulation:
    """Once a simulation is run, it is completely decoupled from the circuit
    which created it. Its pins, while bearing the same name, are unique
    objects.

    Attributes
    ----------
    circuit : simphony.netlist.Subcircuit
        A simulation is instantiated with a completed circuit.
    """

    def __init__(self, circuit: Subcircuit):
        self.circuit = copy.deepcopy(circuit)


class SweepSimulation(Simulation):
    """A swept simulation.

    Parameters
    ----------
    circuit : Subcircuit
        The circuit to be simulated.
    start : float
        The start wavelength (in meters) or frequency (in Hz).
    stop : float
        The stop wavelength (in meters) or frequency (in Hz).
    num : int, optional
        The number of sampled points.
    mode : str, optional
        Defines sweep range mode; either 'wl' for wavelength (m) or
        'freq' for frequency (Hz).

    Attributes
    ----------
    freq : np.ndarray
        The frequency array over which the simulation is performed.
    """

    def __init__(
        self,
        circuit: Subcircuit,
        start: float = 1.5e-6,
        stop: float = 1.6e-6,
        num: int = 2000,
        mode="wl",
    ):
        super().__init__(circuit)
        if start > stop:
            raise ValueError("simulation 'start' value must be less than 'stop' value.")
        if mode == "wl":
            tmp_start = start
            tmp_stop = stop
            start = wl2freq(tmp_stop)
            stop = wl2freq(tmp_start)
        elif mode == "freq":
            pass
        else:
            err = "mode '{}' is not one of 'freq' or 'wl'".format(mode)
            raise ValueError(err)
        if start > stop:
            raise ValueError(
                "starting frequency cannot be greater than stopping frequency"
            )
        self.freq = np.linspace(start, stop, num)
        self.wl = freq2wl(self.freq)

    def simulate(self):
        """Runs the simulation on the object's circuit.

        Returns
        -------
        sim : SweepSimulationResult
            A loaded SweepSimulationResult object.
        """
        models = self._collect_models(self.circuit)
        self.validate_models(models, self.freq)
        cache = self._cache_elements(models, self.freq)

        smat = self._simulate_helper(self.circuit, cache)
        sim = SweepSimulationResult(self.freq, smat)
        return sim

    @staticmethod
    def _collect_models(circuit: Subcircuit, collection: set = set()):
        """Recursively collects all models in a subcircuit.

        Parameters
        ----------
        circuit : simphony.netlist.Subcircuit
            The circuit containing elements to be collected.
        collection : set, optional
            An set for containing simulated models (default is an empty set).

        Returns
        -------
        collection : set
            A set containing all unique models in the circuit.
        """
        # For every item in the circuit
        for item in circuit.elements:

            # If it's an Element type, cache it.
            if issubclass(type(item), Element):
                collection.add(item.model)

            # If it's a subcircuit, recursively call this function.
            elif type(item) is Subcircuit:
                SweepSimulation._collect_models(item, collection)

            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                raise TypeError(
                    'Invalid object in circuit (type "{}")'.format(type(item))
                )

        return collection

    @staticmethod
    def validate_models(models, freq):
        """Ensures all models are valid over the specified frequency range.

        Parameters
        ----------
        models : list
            A list of the model objects to be verified.
        freq : np.ndarray
            The array of frequency values the simulation is defined over.

        Raises
        ------
        NotImplementedError
            If a model does not have a class attribute ``freq_range`` defining
            the valid frequency range for the model.
        ValueError
            If the simulation frequencies are outside of the range of the valid
            frequencies for a model.
        """
        for model in models:
            # Ensure that models have required attributes.
            try:
                lower = min(model.freq_range)
                upper = max(model.freq_range)
            except TypeError:
                raise NotImplementedError(
                    'Does the model "{}" define a valid frequency range?'.format(
                        type(model).__name__
                    )
                )

            # Ensure that models are valid with current simulation parameters.
            if lower > min(freq) or upper < max(freq):
                wmax = freq2wl(max(freq))
                wmin = freq2wl(min(freq))
                wmax_model = freq2wl(upper)
                wmin_model = freq2wl(lower)
                raise ValueError(
                    f'Simulation wavelengths ({wmin} - {wmax}) out of valid bounds for "{model} {wmin_model}-{wmax_model}"'
                )

    @staticmethod
    def _cache_elements(collection, freq):
        cache = {}
        for model in collection:
            cache[model] = model.s_parameters(freq)
        return cache

    @staticmethod
    def _simulate_helper(circuit: Subcircuit, cache: dict):
        netlist = circuit.netlist

        # For every item in the circuit
        for item in circuit.elements:

            # If it's an Element type, simulate it.
            if issubclass(type(item), Element):
                SweepSimulation._create_simulated_result(item, cache)

            # If it's a subcircuit, recursively call this function.
            elif type(item) is Subcircuit:
                SweepSimulation._simulate_helper(item, cache)

            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                err = 'Invalid object in circuit (type "{}")'.format(type(item))
                raise TypeError(err)

        # Connect all the elements together and return a super element.
        built = SweepSimulation.connect_circuit(netlist)
        return built

    @staticmethod
    def _create_simulated_result(element, cache: dict):
        s = cache[element.model]
        sim = ScatteringMatrix(s=s, pinlist=element.pins)
        return sim

    @staticmethod
    def connect_circuit(netlist):
        """Connects the s-matrices of a photonic circuit given its Netlist and
        returns a single 'SimulatedComponent' object containing the frequency
        array, the assembled s-matrix, and a list of the external nets
        (negative integers).

        Parameters
        ----------
        component_list : List[SimulatedComponent]
            A list of the components to be connected.
        net_count : int
            The total number of internal nets in the component list.

        Returns
        -------
        combined : ScatteringMatrix
            After the circuit has been fully connected, the result is a single
            ComponentSimulation with fields f (frequency), s (s-matrix), and nets
            (external ports: negative numbers, as strings).

        Notes
        -----
        This function doesn't actually store ``combined`` on each iteration
        through the netlist. That's because the Pin objects can only reference
        one PinList at a time, which in turn can only reference one Element.
        Since we transferring the actual Pin objects between lists, keeping
        a reference to the Pin also keeps a reference to the ``combined``
        Element alive. Hence, we track pins but not the ``SimulationResult``.
        """
        _logger = _module_logger.getChild("SweepSimulation.connect_circuit")

        # FIXME: What if there are no items in the netlist (only one element
        # in the circuit)?
        for net in netlist:
            p1, p2 = net
            if p1.element == p2.element:
                _logger.debug("Internal connection")
                combined = ScatteringMatrix()
                combined.s = innerconnect_s(p1.element.s, p1.index, p2.index)
                pinlist = p1.pinlist
                pinlist.remove(p1, p2)
                combined.pinlist = pinlist
            else:
                _logger.debug("External connection")
                combined = ScatteringMatrix()
                combined.s = connect_s(p1.element.s, p1.index, p2.element.s, p2.index)
                pinlist = p1.pinlist + p2.pinlist
                pinlist.remove(p1, p2)
                combined.pinlist = pinlist
        return combined


class MonteCarloSweepSimulation(SweepSimulation):
    """A monte carlo sweep simulation.

    Parameters
    ----------
    circuit : Subcircuit
        The circuit to be simulated.
    start : float
        The start wavelength (in meters) or frequency (in Hz).
    stop : float
        The stop wavelength (in meters) or frequency (in Hz).
    num : int
        The number of sampled points.
    mode : str
        Defines sweep range mode; either 'wl' for wavelength (m) or
        'freq' for frequency (Hz).
    """

    def __init__(
        self,
        circuit: Subcircuit,
        start: float = 1.5e-6,
        stop: float = 1.6e-6,
        num: int = 2000,
        mode="wl",
    ):
        super().__init__(circuit, start, stop, num, mode)

    def simulate(self, runs=10):
        """
        Parameters
        ----------
        runs : int, optional
            The number of monte carlo iterations to run (default 10).

        Returns
        -------
        sim : MonteCarloSimulationResult
            A loaded MonteCarloSimulationResult object.
        """
        models = SweepSimulation._collect_models(self.circuit)
        SweepSimulation.validate_models(models, self.freq)
        cache = SweepSimulation._cache_elements(models, self.freq)

        smat = self._simulate_helper(copy.deepcopy(self.circuit), cache)
        sim = MonteCarloSimulationResult(self.freq, smat, runs)

        for run in range(runs):
            for model in models:
                model.regenerate_monte_carlo_parameters()
            cache = self._cache_elements_monte_carlo(models, self.freq)
            sim.add_result(self._simulate_helper(copy.deepcopy(self.circuit), cache))

        return sim

    @staticmethod
    def _cache_elements_monte_carlo(collection, freq):
        cache = {}
        for model in collection:
            cache[model] = model.monte_carlo_s_parameters(freq)
        return cache
