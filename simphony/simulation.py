# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.simulation
===================

This module contains the simulation context as well as simulation devices to be
used within the context. Devices include theoretical sources and detectors.
"""

from cmath import rect
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple

import numpy as np
import scipy
from scipy.constants import epsilon_0, h, mu_0
from scipy.signal import butter, sosfiltfilt

from simphony import Model
from simphony.tools import wl2freq

if TYPE_CHECKING:
    from simphony.layout import Circuit


# this variable keeps track of the current simulation context (if any)
context = None

# create an np compatible rect function
nprect = np.vectorize(rect)


def from_db(x: float, factor: int = 10) -> float:
    """Converts a dB value to a linear value."""
    return 10 ** (x / factor)


def to_db(x: float, factor: int = 10) -> float:
    """Converts a linear value to a dB value."""
    return factor * np.log10(x)


class Simulation:
    """This class instantiates a simulation context.

    Any simulation devices that are instantiated within the context
    block are managed by the instance of this class.
    """

    circuit: ClassVar["Circuit"]
    detectors: ClassVar[List["Detector"]]
    sources: ClassVar[List["Source"]]
    s_parameters_method: ClassVar[str]

    def __enter__(self) -> "Simulation":
        # set this as the global context
        self.set_context(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # remove all devices from the circuit
        for detector in self.detectors:
            detector.disconnect()

        for source in self.sources:
            source.disconnect()

        self.circuit = None

        # remove this as the global context
        self.set_context(None)

        return False

    def __init__(self, *, fs: float = 1e9, seed: Optional[int] = None) -> None:
        """Initializes the simulation context.

        Parameters
        ----------
        fs :
            The sampling frequency to simulate with.
        seed :
            The seed for the RNG for the sampling.
        """
        self.circuit = None
        self.detectors = []
        self.eta = np.sqrt(mu_0 / (epsilon_0 * 11.68))  # 11.68 is silicon's
        self.freqs = np.array([])
        self.fs = fs
        self.noise = False
        self.num_samples = 1
        self.rng = np.random.default_rng(seed)
        self.powers = np.array([])
        self.shape = [0, 0]
        self.sources = []
        self.s_params = np.array([])
        self.s_parameters_method = "s_parameters"
        self.transmissions = []

    def _add_detector(self, detector: "Detector") -> None:
        """Adds a detector to the simulation context.

        Parameters
        ----------
        detector :
            The detector to add to the context.
        """
        self.detectors.append(detector)

    def _add_source(self, source: "Source") -> None:
        """Adds a source to the simulation context.

        Parameters
        ----------
        source :
            The source to add to the context.
        """
        self.sources.append(source)

    def _expand_array(self, arr: np.array, size: int) -> np.array:
        """Returns an expanded version of the given array.

        Example:
            arr = [1, 2, 3]
            size = 5
            returns [1, 1, 2, 2, 3]

        Parameters
        ----------
        arr :
            The array to expand
        size :
            The length to expand the array to
        """
        # only expand the array if it's smaller than the given size
        arr_len = len(arr)
        if arr_len >= size:
            return arr

        # calculate how many times each value needs to be repeated
        expanded = np.zeros(size)
        repeat = int(size / arr_len)
        remainder = size % arr_len

        # expand each value in the given array
        for i, value in enumerate(arr):
            # calculate ranges, accounting for remainders
            # end range has +1 because it is non-inclusive
            start = i * repeat + min(i, remainder)
            end = start + repeat + 1 + (1 if i < remainder else 0)

            # put the values into the expanded array
            expanded[start:end] = value

        return expanded

    def _get_signals(self) -> np.ndarray:
        """Get the signals in the order set by the detectors. Each signal is a
        multi-dimensional array. The first index corresponds to frequency. The
        second index corresponds to power. The third index corresponds to
        sample number. For example, ``signal[freq][power][sample]``.

        This method returns an array of signals if there are multiple,
        or a single signal if there is only one.
        """
        # make sure we have detectors and sources connected
        if len(self.detectors) == 0 or len(self.sources) == 0:
            raise RuntimeError(
                "At least one `Detector` and `Source` needs to be connected to the circuit."
            )

        # figure out the frequencies and powers to use during simulation
        # we will use the intersection of frequencies between sources
        # and use the individual powers defined by the sources
        freqs = None
        self.shape = [0, 0]
        for source in self.sources:
            # take the intersection between frequencies
            freqs = (
                source._freqs if freqs is None else np.intersect1d(freqs, source._freqs)
            )
            self.shape[0] = len(freqs)

            # for now, just keep track of the biggest power
            if len(source._powers) > self.shape[1]:
                self.shape[1] = len(source._powers)

        self.freqs = freqs

        # now that frequencies are calculated, have the sources load the data
        for source in self.sources:
            source._load_context_data()

            # keep track of which pin the source is connected to
            source.index = self.circuit.get_pin_index(source.pins[0]._connection)

        # get the scattering parameters
        self.s_params = self.s_parameters(self.freqs)

        # construct the signals determined by the detectors
        signals = []
        for detector in self.detectors:
            # calculate the power detected at each detector pin
            powers = []
            for pin in detector.pins:
                output_index = self.circuit.get_pin_index(pin._connection)

                # figure out how the sources interfere
                transmissions = 0
                for i, pin in enumerate(self.circuit.pins):
                    # calculate transmissions for every source connected to
                    # the circuit
                    for source in self.sources:
                        if source.index == i:
                            break
                    else:
                        continue

                    # calculate how much this source contributes to the output field
                    scattering = self.s_params[:, output_index, source.index]
                    contributions = scattering[:, np.newaxis] * nprect(
                        np.sqrt(source._coupled_powers * 2 * self.eta),
                        source.phase,
                    )

                    # add all of the different source contributions together
                    contributions = contributions[:, :, np.newaxis] + np.zeros(
                        self.num_samples
                    )
                    transmissions += contributions

                # convert the output fields to powers
                self.transmissions.append(transmissions)
                powers.append((np.abs(transmissions) ** 2 / (2 * self.eta)))

            # send the powers through the detectors to convert to signals
            signals.extend(detector._detect(powers))

        # if there's only one signal, don't return it in an array
        signals = np.array(signals)
        if len(signals) == 1:
            return signals[0]

        return signals

    def monte_carlo(self, flag: bool) -> None:
        """Sets whether or not to use the Monte Carlo scattering parameters.

        Parameters
        ----------
        flag :
            When True, Monte Carlo scattering parameters will be used. When
            False, they will not be used.
        """
        self.s_parameters_method = (
            "monte_carlo_s_parameters" if flag else "s_parameters"
        )

    def _compute_correlated_samples(self, coords, sigmaw, sigmat, l, runs):
        x = [0] * len(coords)
        y = [0] * len(coords)

        # get x and y co-ordinates
        components = self.circuit._get_components()[:-2]
        if len(coords) != len(components):
            raise ValueError('Incorrect number of component coordinates passed to argument "coords".')

        for k, v in coords.items():
            if k in components:
                x[components.index(k)] = v['x']
                y[components.index(k)] = v['y']
            else:
                raise KeyError(f'Component {k} not in circuit.')

        n = len(self.circuit._get_components()) - 2
        corr_matrix_w = np.zeros((n, n))
        corr_matrix_t = np.zeros((n, n))

        # generate correlation values
        for i in range(n):
            for k in range(n):

                corr_val = np.exp(- ((x[k] - x[i]) ** 2 + (y[k] - y[i]) ** 2) / (0.5 * (l ** 2)))

                corr_matrix_w[i][k] = corr_matrix_w[k][i] = corr_val
                corr_matrix_t[i][k] = corr_matrix_t[k][i] = corr_val

        cov_matrix_w = np.zeros((n, n))
        cov_matrix_t = np.zeros((n, n))
        # generate covariance matrix
        for i in range(n):
            for k in range(n):
                cov_matrix_w[i][k] = sigmaw * corr_matrix_w[i][k] * sigmaw
                cov_matrix_t[i][k] = sigmat * corr_matrix_t[i][k] * sigmat

        try:
            # perform Cholesky decomposition on the covariance matrices
            l_w = scipy.linalg.cholesky(cov_matrix_w, lower=True)
            l_t = scipy.linalg.cholesky(cov_matrix_t, lower=True)
        except np.linalg.LinAlgError:
            # if matrix is not positive-definite, do LU decomposition
            _, l_w, _ = scipy.linalg.lu(cov_matrix_w)
            _, l_t, _ = scipy.linalg.lu(cov_matrix_t)

        # generate random distribution with mean 0 and standard deviation of 1, size: no. of elements x no. of runs
        X = np.random.multivariate_normal(np.zeros(n), np.eye(n, n), runs).T

        # generate correlation samples
        corr_sample_matrix_w = np.dot(l_w, X)
        corr_sample_matrix_t = np.dot(l_t, X)
        return corr_sample_matrix_w, corr_sample_matrix_t

    def layout_aware_simulation(self, coords: dict = {}, sigmaw: float = 5, sigmat: float = 2, l: float = 4.5e-3, runs: int = 10, num_samples: int = 1, **kwargs) -> Tuple[np.array, np.array]:
        """Runs the layout-aware Monte Carlo sweep simulation for the circuit.

        Parameters
        ----------
        coords :
            Dictionary of co-ordinates. Each item in the dict has a component in the circuit as
            a key and its x and y co-ordinates as values
        sigmaw :
            Standard deviation of width variations (default 5)
        sigmat :
            Standard deviation of thickness variations (default 2)
        l :
            Correlation length (m) (default 4.5e-3)
        dB :
            Returns the power ratios in deciBels when True.
        mode :
            Whether to return frequencies or wavelengths for the corresponding
            power ratios. Defaults to whatever values were passed in upon
            instantiation.
        runs :
            The number of Monte Carlo iterations to run (default 10).
        """
        results = []
        corr_sample_matrix_w, corr_sample_matrix_t = self._compute_correlated_samples(coords, sigmaw, sigmat, l, runs)
        components = self.circuit._get_components()
        n = len(components) - 2
        for i in range(runs):
            # use s_parameters for the first run, then monte_carlo_* for the rest

            for idx in range(n):

                # update component parameters
                components[idx].update_variations(corr_w=corr_sample_matrix_w[idx][i], corr_t=corr_sample_matrix_t[idx][i])

            self.s_parameters_method = "s_parameters" if i == 0 else "layout_aware_monte_carlo_s_parameters"
            print(f'Run {i} of {runs}')
            results.append(
                self.sample(num_samples=num_samples)
            )

            for idx in range(n):
                components[idx].regenerate_layout_aware_monte_carlo_parameters()

        return results

    def s_parameters(self, freqs: np.array) -> np.ndarray:
        """Gets the scattering parameters for the specified frequencies.

        Parameters
        ----------
        freqs :
            The list of frequencies to run simulations for.
        """
        # make sure we have a circuit ready to simulate
        if self.circuit is None:
            raise RuntimeError(
                "At least one `Detector` or `Source` needs to be connected to the circuit."
            )

        # get the scattering parameters
        subcircuit = self.circuit.to_subcircuit(permanent=False)

        # ensure valid frequency
        lower, upper = subcircuit.freq_range
        if lower > freqs[0] or upper < freqs[-1]:
            raise ValueError(
                f"Cannot simulate the range ({freqs[0], freqs[-1]}) over the valid range ({lower}, {upper})"
            )

        # get the scattering parameters for the specified method and frequencies
        return getattr(subcircuit, self.s_parameters_method)(freqs)

    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Samples the outputs of the circuit. If more than one sample is
        requested, noise will be injected into the system. If only one sample
        is requested, the returned value will be purely theoretical.

        Parameters
        ----------
        num_samples :
            The number of samples to take. If only one sample is taken, it will
            be the theoretical value of the circuit. If more than one sample is
            taken, they will vary based on simulated noise.
        """
        # filtering requires an odd number of samples.
        # if they want an even number, we will add one and then remove a
        # sample at the end
        _num_samples = num_samples
        if num_samples % 2 == 0:
            _num_samples += 1

        # if we are taking more than one sample, include noise
        self.num_samples = _num_samples
        self.noise = self.num_samples > 1

        # sample the signals
        signals = self._get_signals()

        # remove the extra sample if we added one
        if _num_samples != num_samples:
            return signals[:, :, :num_samples]
        else:
            return signals

    @classmethod
    def get_context(cls) -> "Simulation":
        """Gets the current simulation context."""
        global context
        return context

    @classmethod
    def set_context(cls, _context: "Simulation") -> None:
        """Sets the current simulation context.

        Parameters
        ----------
        _context :
            The current ``Simulation`` instance.
        """
        global context
        context = _context


class SimulationModel(Model):
    """A Simphony model that is aware of the current Simulation context.

    Models that extend this one should automatically connect to the
    context upon instantiation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context = Simulation.get_context()

    def _on_connect(self, *args, **kwargs):
        super()._on_connect(*args, **kwargs)

        # after this model connects to another model, we have access to the
        # circuit. make the context aware of the circuit
        self.context.circuit = self.circuit


class Source(SimulationModel):
    """A simphony model for a source.

    It automatically connects to the current simulation context upon
    instantiation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context._add_source(self)

    def _load_context_data(self) -> None:
        """Gets the frequencies and powers to sweep from the simulation
        context.

        This information must be updated so that the simulated data all
        has the same shape.
        """
        self.freqs = self.context.freqs
        self.powers = self.context._expand_array(self._powers, self.context.shape[1])
        self._coupled_powers = self.powers * self.coupling_ratio


class Laser(Source):
    """A Simphony model for a laser source."""

    pin_count = 1

    def __init__(
        self,
        *args,
        coupling_loss=0,
        freq=None,
        phase=0,
        power=0,
        rin=-np.inf,
        wl=1550e-9,
        **kwargs,
    ) -> None:
        """Initializes the laser.

        Parameters
        ----------
        coupling_loss :
            The coupling loss of the laser in dB.
        freq :
            The frequency of the laser in Hz.
        phase :
            The phase of the laser in radians.
        power :
            The power of the laser in Watts.
        rin :
            The relative intensity noise of the laser in dBc/Hz.
        wl :
            The wavelength of the laser in meters.
        """
        super().__init__(*args, **kwargs)

        # initialize properties
        self._coupled_powers = np.array([])
        self._freqs = np.array([freq if freq else wl2freq(wl)])
        self._powers = np.array([power])
        self.coupling_ratio = from_db(-np.abs(coupling_loss))
        self.freqs = np.array([])
        self.index = 0
        self.phase = phase
        self.powers = np.array([])
        self.rin = -np.abs(rin)
        self.rin_dists = {}

    def freqsweep(self, start: float, end: float, num: int = 500) -> "Laser":
        """Sets the frequencies to sweep during simulation.

        Parameters
        ----------
        start :
            The frequency to start at.
        end :
            The frequency to end at.
        num :
            The number of frequencies to sweep.
        """
        self._freqs = np.linspace(start, end, num)
        return self

    def get_rin(self, bandwidth: float) -> float:
        """Gets the RIN value in dBc.

        Parameters
        ----------
        bandwidth :
            The bandwidth of the detector in Hz.
        """
        return 0 if self.rin == -np.inf else to_db(from_db(self.rin) * bandwidth)

    def get_rin_dist(self, i: int, j: int) -> List[float]:
        """Returns the normal distribution used for the i,j key. If this is the
        first access, a new distribution is created.

        Parameters
        ----------
        i :
            The first index (frequency index).
        j :
            The second index (power index).
        """
        try:
            return self.rin_dists[i, j]
        except KeyError:
            self.rin_dists[i, j] = self.context.rng.normal(
                size=self.context.num_samples
            )
            return self.rin_dists[i, j]

    def powersweep(self, start: float, end: float, num: int = 500) -> "Laser":
        """Sets the powers to sweep during simulation.

        Parameters
        ----------
        start :
            The power to start at.
        end :
            The power to end at.
        num :
            The number of powers to sweep.
        """
        self._powers = np.linspace(start, end, num)
        return self

    def wlsweep(self, start: float, end: float, num: int = 500) -> "Laser":
        """Sets the wavelengths to sweep during simulation.

        Parameters
        ----------
        start :
            The wavelength to start at.
        end :
            The wavelength to end at.
        num :
            The number of wavelengths to sweep.
        """
        self._freqs = wl2freq(np.linspace(start, end, num))[::-1]
        return self


class Detector(SimulationModel):
    """The base class for all detectors.

    When a detector is connected to the circuit, it defines how many
    outputs are returned from calling the ``Simulation.sample`` method.
    This detector only adds one output.
    """

    pin_count = 1

    def __init__(
        self,
        *args,
        conversion_gain=1,
        high_fc=1e9,
        low_fc=0,
        noise=0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.context._add_detector(self)

        # conversion gain = responsivity * transimpedance gain
        # noise = Vrms on measurement
        self.conversion_gain = conversion_gain
        self.high_fc = high_fc
        self.low_fc = low_fc
        self.noise = noise
        self.noise_dists = np.array([])
        self.rin_dists = np.array([])

    def _detect(self, power: List[np.ndarray]) -> List[np.ndarray]:
        """This method receives the signal values as powers, i.e. the units are
        in Watts.

        Other detectors should extend this method to inject noise,
        amplify the signal, etc.
        """
        # temporarily unwrap the signal for easier manipulation
        power = power[0]

        # prepare noise dists with the proper shape
        self.noise_dists = np.zeros(power.shape)
        self.rin_dists = np.zeros(power.shape)

        if self.context.noise:
            # for now, we assume that all sources are lasers and inject
            # quantum noise using poissonian distributions

            # inject an independent distribution for each frequency and power
            for i, freq in enumerate(self.context.freqs):
                # power = num_photons * h * freq * sampling_freq
                hffs = h * freq * self.context.fs
                for j, _ in enumerate(power[i]):
                    # calulcate the RIN noise
                    noise = np.zeros(self.context.num_samples)
                    for source in self.context.sources:
                        # we let the laser handle the RIN distribution
                        # so the same noise is injected in all the signals
                        rin = source.get_rin(self._get_bandwidth()[0])
                        dist = source.get_rin_dist(i, j)

                        # calculate the standard deviation of the RIN noise
                        std = (
                            from_db(to_db(power[i][j][0]) + rin, 20)
                            if power[i][j][0]
                            else 0
                        )

                        noise += std * dist

                    # store the RIN noise for later use
                    self.rin_dists[i][j] = noise

                    # calculate and store electrical noise for later use
                    self.noise_dists[i][j] = self.noise * self.context.rng.normal(
                        size=self.context.num_samples
                    )

                    # power[i][j] has the correct shape but all of the values
                    # are the raw power. so we get one of those values and
                    # calculate the corresponding photon number. we then
                    # take a photon number distribution and convert it to power
                    # which we then use as our samples.
                    power[i][j] = hffs * self.context.rng.poisson(
                        power[i][j][0] / hffs, self.context.num_samples
                    )

        # amplify the signal
        signal = (power + self.rin_dists) * self.conversion_gain

        # add electrical noise on top
        signal += self.noise_dists

        # filter the signal
        if self.context.num_samples > 1:
            signal = self._filter(signal)

        # wrap the signal back up
        return np.array([signal])

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        """Filters the signal.

        Parameters
        ----------
        signal :
            The signal to filter.
        """
        bw = self._get_bandwidth()
        sos = (
            butter(6, bw[2], "lowpass", fs=self.context.fs, output="sos")
            if bw[1] == 0
            else butter(
                6,
                [
                    bw[1],
                    bw[2],
                ],
                "bandpass",
                fs=self.context.fs,
                output="sos",
            )
        )
        return sosfiltfilt(sos, signal)

    def _get_bandwidth(self):
        """Gets the bandwidth of the detector w.r.t.

        the sampling frequency.
        """
        low = (
            0
            if self.low_fc == 0
            else max(self.low_fc, self.context.fs / self.context.num_samples * 30)
        )
        high = min(self.high_fc, 0.5 * (self.context.fs - 1))

        return (high - low, low, high)


class DifferentialDetector(Detector):
    """A differential detector takes two connections and provides three outputs
    to the ``Simulation.sample`` method.

    The outputs are [connection1, connection1 - connection2, connection2]. The
    first and third outputs are the monitor outputs and the second output is the
    RF output.
    """

    pin_count = 2

    def __init__(
        self,
        *args,
        monitor_conversion_gain=1,
        monitor_high_fc=1e9,
        monitor_low_fc=0,
        monitor_noise=0,
        rf_cmrr=np.inf,
        rf_conversion_gain=1,
        rf_high_fc=1e9,
        rf_low_fc=0,
        rf_noise=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # initialize parameters
        self.monitor_conversion_gain = monitor_conversion_gain
        self.monitor_high_fc = monitor_high_fc
        self.monitor_low_fc = monitor_low_fc
        self.monitor_noise = monitor_noise
        self.monitor_noise_dists = np.array([])
        self.monitor_rin_dists1 = np.array([])
        self.monitor_rin_dists2 = np.array([])
        self.rf_cmrr = -np.abs(rf_cmrr)
        self.rf_conversion_gain = rf_conversion_gain
        self.rf_high_fc = rf_high_fc
        self.rf_low_fc = rf_low_fc
        self.rf_noise = rf_noise
        self.rf_noise_dists = np.array([])
        self.rf_rin_dists1 = np.array([])
        self.rf_rin_dists2 = np.array([])

    def _detect(self, powers: List[np.ndarray]) -> List[np.ndarray]:
        p1 = powers[0]
        p2 = powers[1]

        # prepare noise dists with the proper shape
        self.monitor_noise_dists = np.zeros(p1.shape)
        self.monitor_rin_dists1 = np.zeros(p1.shape)
        self.monitor_rin_dists2 = np.zeros(p2.shape)
        self.rf_noise_dists = np.zeros(p1.shape)
        self.rf_rin_dists1 = np.zeros(p1.shape)
        self.rf_rin_dists2 = np.zeros(p2.shape)

        if self.context.noise:
            # for now, we assume that all sources are lasers and inject
            # quantum noise using poissonian distributions

            # inject an independent distribution for each frequency and power
            for i, freq in enumerate(self.context.freqs):
                # power = num_photons * h * freq * sampling_freq
                hffs = h * freq * self.context.fs
                for j, _ in enumerate(p1[i]):
                    # calculate the RIN noise
                    # we will need to keep track of the noise for each
                    # differential signal independently
                    monitor_noise1 = np.zeros(self.context.num_samples)
                    monitor_noise2 = np.zeros(self.context.num_samples)
                    rf_noise1 = np.zeros(self.context.num_samples)
                    rf_noise2 = np.zeros(self.context.num_samples)

                    # every source will contribute different noise
                    for source in self.context.sources:
                        # get the RIN specs from the laser to ensure that the
                        # same noise is injected across all signals
                        monitor_rin = source.get_rin(self._get_monitor_bandwidth()[0])
                        rf_rin = source.get_rin(self._get_rf_bandwidth()[0])
                        dist = source.get_rin_dist(i, j)

                        # if the two powers are different, we need to adjust
                        # the CMRR to account for the difference
                        cmrr = self.rf_cmrr
                        sum = p1[i][j][0] + p2[i][j][0]
                        diff = np.abs(p1[i][j][0] - p2[i][j][0])
                        if diff:
                            cmrr2 = -to_db(sum / diff)
                            cmrr = to_db(
                                np.sqrt(from_db(cmrr) ** 2 + from_db(cmrr2) ** 2)
                            )

                        # only calculate the noise if there is power
                        if p1[i][j][0] > 0:
                            p1db = to_db(p1[i][j][0])
                            monitor_noise1 += from_db(p1db + monitor_rin, 20) * dist
                            rf_noise1 += from_db(p1db + rf_rin + cmrr, 20) * dist

                        # only calculate the noise if there is power
                        if p2[i][j][0] > 0:
                            p2db = to_db(p2[i][j][0])
                            monitor_noise2 += from_db(p2db + monitor_rin, 20) * dist
                            rf_noise2 += from_db(p2db + rf_rin + cmrr, 20) * dist

                    # store the RIN noise for later use
                    self.monitor_rin_dists1[i][j] = monitor_noise1
                    self.monitor_rin_dists2[i][j] = monitor_noise2
                    self.rf_rin_dists1[i][j] = rf_noise1
                    self.rf_rin_dists2[i][j] = rf_noise2

                    # calculate and store electrical noise
                    self.monitor_noise_dists[i][
                        j
                    ] = self.monitor_noise * self.context.rng.normal(
                        size=self.context.num_samples
                    )
                    self.rf_noise_dists[i][j] = self.rf_noise * self.context.rng.normal(
                        size=self.context.num_samples
                    )

                    # p1[i][j] has the correct shape but all of the values
                    # are the raw power. so we get one of those values and
                    # calculate the corresponding photon number. we then
                    # take a photon number distribution and convert it to power
                    # which we then use as our samples.
                    p1[i][j] = hffs * self.context.rng.poisson(
                        p1[i][j][0] / hffs, self.context.num_samples
                    )

                    # we do the same for the second signal
                    p2[i][j] = hffs * self.context.rng.poisson(
                        p2[i][j][0] / hffs, self.context.num_samples
                    )

        # return the outputs
        return (
            self._monitor(p1, self.monitor_rin_dists1),
            self._rf(p1, p2),
            self._monitor(p2, self.monitor_rin_dists2),
        )

    def _filter(self, signal: np.ndarray, bw: Tuple[float, float, float]) -> np.ndarray:
        """Filters the signal.

        Parameters
        ----------
        signal :
            The signal to filter.
        bw :
            The bandwidth of the filter. (difference, low_fc, high_fc)
        """
        sos = (
            butter(6, bw[2], "lowpass", fs=self.context.fs, output="sos")
            if bw[1] == 0
            else butter(
                6,
                [
                    bw[1],
                    bw[2],
                ],
                "bandpass",
                fs=self.context.fs,
                output="sos",
            )
        )
        return sosfiltfilt(sos, signal)

    def _get_bandwidth(self, low_fc, high_fc):
        """Gets the bandwidth of the detector w.r.t. the sampling frequency.

        Parameters
        ----------
        low_fc :
            The lower cut-off frequency.
        high_fc :
            The higher cut-off frequency.
        """
        low = (
            0
            if low_fc == 0
            else max(low_fc, self.context.fs / self.context.num_samples * 30)
        )
        high = min(high_fc, 0.5 * (self.context.fs - 1))

        return (high - low, low, high)

    def _get_monitor_bandwidth(self):
        """Gets the bandwidth of the monitor w.r.t.

        the sampling frequency.
        """
        return self._get_bandwidth(self.monitor_low_fc, self.monitor_high_fc)

    def _get_rf_bandwidth(self):
        """Gets the bandwidth of the rf w.r.t.

        the sampling frequency.
        """
        return self._get_bandwidth(self.rf_low_fc, self.rf_high_fc)

    def _monitor(self, power: np.ndarray, rin_dists: np.ndarray) -> np.ndarray:
        """Takes a signal and turns it into a monitor output.

        Parameters
        ----------
        power :
            The power to convert to a monitor signal.
        """
        # amplify the signal
        signal = (power + rin_dists) * self.monitor_conversion_gain

        # add the electrical noise after amplification
        signal += self.monitor_noise_dists

        # filter the signal
        if self.context.num_samples > 1:
            signal = self._monitor_filter(signal)

        return signal

    def _monitor_filter(self, signal: np.ndarray) -> np.ndarray:
        """Filters the monitor signal.

        Parameters
        ----------
        signal :
            The signal to filter.
        """
        return self._filter(signal, self._get_monitor_bandwidth())

    def _rf(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Takes two signals and generates the differential RF signal. p1 - p2.

        Parameters
        ----------
        p1 :
            The first signal (in Watts).
        p2 :
            The second signal (in Watts)."""
        # amplify the difference.
        # we don't subtract RIN dists because that would be a CMRR of infinity.
        # when we generated the RIN, we took the CMRR into account so at this
        # point, all we need to do is add them after the quantum signals have
        # been diffed.
        signal = (
            (p1 - p2) + (self.rf_rin_dists1 + self.rf_rin_dists2)
        ) * self.rf_conversion_gain

        # add the electrical signal after amplification
        signal += self.rf_noise_dists

        # filter the difference
        if self.context.num_samples > 1:
            signal = self._rf_filter(signal)

        return signal

    def _rf_filter(self, signal: np.ndarray) -> np.ndarray:
        """Filters the RF signal.

        Parameters
        ----------
        signal :
            The signal to filter.
        """
        return self._filter(signal, self._get_rf_bandwidth())
