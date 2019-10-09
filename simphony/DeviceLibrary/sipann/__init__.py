import simphony.core as core
from simphony.core import register_component_model

import os
import numpy as np
from itertools import combinations_with_replacement as comb_w_r
from scipy import special
from SiPANN import dc

from numba import njit
import ctypes
from numba.extending import get_cython_function_address

@register_component_model
class sipann_wg_integral(core.ComponentModel):
    """Neural-net trained model of a waveguide.
    """
    ports = 2
    cachable = False

    # TODO: Remove the delta_length part of this model; should be implemented
    # only in the simphony.simulation.MonteCarloSimulation part of the program
    @classmethod
    def s_parameters(cls,
                    length: float=0,
                    width: float=0.5,
                    thickness: float=0.22,
                    radius: float=5,
                    delta_length: float=0,
                    points: list = [],
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.

        Parameters
        ----------
        length: float
            Length of the waveguide.
        width: float
            Width of the waveguide in microns.
        thickness: float
            Thickness of the waveguide in microns.
        radius: float
            Bend radius of bends in the waveguide.
        delta_length: float
            Only used in monte carlo simulations to randomly vary length.
        points: list
            The points denoting the path of the waveguide.
        start_freq: float
            The starting frequency to obtain s-parameters for.
        stop_freq: float
            The ending frequency to obtain s-parameters for.
        num: int
            The number of points to use between start_freq and stop_freq.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`.
        """
        frequency = np.linspace(start_freq, stop_freq, num)
        return cls.ann_s_params(frequency, length, width, thickness, delta_length)

    @staticmethod
    def cartesian_product(arrays):
        la = len(arrays)
        dtype = np.find_common_type([a.dtype for a in arrays], [])
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def straightWaveguide(wavelength, width, thickness, angle):
        # Sanitize the input
        if type(wavelength) is np.ndarray:
            wavelength = np.squeeze(wavelength)
        else:
            wavelength = np.array([wavelength])
        if type(width) is np.ndarray:
            width = np.squeeze(width)
        else:
            width = np.array([width])
        if type(thickness) is np.ndarray:
            thickness = np.squeeze(thickness)
        else:
            thickness = np.array([thickness])
        if type(angle) is np.ndarray:
            angle = np.squeeze(angle)
        else:
            angle = np.array([angle])

        INPUT  = ann_wg_integral.cartesian_product([wavelength,width,thickness,angle])

        #Get all possible combinations to use
        degree = 4
        features = 4
        combos = []
        for i in range(5):
            combos += [k for k in comb_w_r(range(degree),i)]

        #make matrix of all combinations
        n = len(INPUT)
        polyCombos = np.ones((n,len(combos)))
        for j,c in enumerate(combos):
            if c == ():
                polyCombos[:,j] = 1
            else:
                for k in c:
                    polyCombos[:,j] *= INPUT[:,k]

        #get coefficients and return
        coeffs = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_wg_integral_1550', 'straightCoeffs.npy'))
        return polyCombos@coeffs

    @staticmethod
    def ann_s_params(frequency, length: float, width: float, thickness: float, delta_length: float, **kwargs):
        '''
        Function that calculates the s-parameters for a waveguide using the ANN model

        Parameters
        ----------
        frequency : np.array
        length : float
        width : float
        thickness : float
        delta_length : float
        **kwargs : None
            This is a redundancy in case other parameters are included which
            are unnecessary for calculating the result.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`.
        '''

        mat = np.zeros((len(frequency),2,2), dtype=complex)

        c0 = 299792458 #m/s
        mode = 0 #TE
        TE_loss = 700 #dB/m for width 500nm
        alpha = TE_loss/(20*np.log10(np.exp(1))) #assuming lossless waveguide
        waveguideLength = length + (length * delta_length)

        #calculate wavelength
        wl = np.true_divide(c0,frequency)

        # effective index is calculated by the ANN
        neff = ann_wg_integral.straightWaveguide(np.transpose(wl), width, thickness, 90)

        #K is calculated from the effective index and wavelength
        K = (2*np.pi*np.true_divide(neff,wl))

        #the s-matrix is built from alpha, K, and the waveguide length
        for x in range(0, len(neff)):
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*waveguideLength + (K[x]*waveguideLength*1j))
        s = mat

        return (frequency, s)

@register_component_model
class sipann_dc_straight(core.ComponentModel):
    """Regression Based Closed Form solution of a straight directional coupler
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    length: float=0,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        length:     float  Length of the waveguide.
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Gap between the two waveguides
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        length    = length*1000
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)
        item = dc.Straight(width, thickness, gap, length)
        return item.sparams(wl)

@register_component_model
class sipann_dc_halfracetrack(core.ComponentModel):
    """Regression Based Closed Form solution of half a racetrack resonator
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    length: float=0,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    radius: float=10,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        length:     float  Length of the waveguide.
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Gap between the two waveguides
        radius:     float  Radius of bent portions of waveguide
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        length    = length*1000
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000
        radius    = radius*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.Racetrack(width, thickness, radius, gap, length)
        return item.sparams(wl)

@register_component_model
class sipann_dc_halfring(core.ComponentModel):
    """Regression Based Closed Form solution of half of a ring resonator
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    radius: float=10,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Gap between the two waveguides
        radius:     float  Radius of bent portions of waveguide
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000
        radius    = radius*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.RR(width, thickness, radius, gap)
        return item.sparams(wl)


@register_component_model
class sipann_dc_standard(core.ComponentModel):
    """Regression Based Closed Form solution of a standard shaped directional coupler
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    length: float=10,
                    H: float=2,
                    V: float=7,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Gap between the two waveguides
        length:     float  Length of coupling region
        H:          float  Horizontal distance of s-bends on end
        V:          float  Vertical distance of s-bends on end
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000
        length    = length*1000
        H         = H*1000
        V         = V*1000


        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.Standard(width, thickness, gap, length, H, V)
        return item.sparams(wl)


@register_component_model
class sipann_dc_doublehalfring(core.ComponentModel):
    """Regression Based Closed Form solution of double half ring resonator
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    radius: float=10,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Minimum gap between the two waveguides
        radius:     float  Radius of halfrings (on both sides)
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000
        radius    = radius*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.DoubleRR(width, thickness, radius, gap)
        return item.sparams(wl)


@register_component_model
class sipann_dc_angledhalfring(core.ComponentModel):
    """Regression Based Closed Form solution of an angled ring resonator
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    width: float=0.5,
                    thickness: float=0.22,
                    gap: float=0.1,
                    radius: float=10,
                    theta: float=np.pi/4,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        float  Gap between the two waveguides
        radius:     float  Radius of bent portions of waveguide
        theta:      float  Angled distance where through waveguide is parallel to ring
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000
        gap       = gap*1000
        radius    = radius*1000
        theta     = theta*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.AngledRR(width, thickness, radius, gap, theta)
        return item.sparams(wl)

@register_component_model
class sipann_dc_arbitrarysym(core.ComponentModel):
    """Regression Based form of any directional coupler provided gap function
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    gap: callable,
                    dgap: callable,
                    zmin: float,
                    zmax: float,
                    width: float=0.5,
                    thickness: float=0.22,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        function  Gap between the two waveguides as a function of z (in nm for now)
        dgap:       function  Derivative of above gap function (in nm for now)
        zmin:       float  Beginning of dc in gap function (also in nm)
        zmax:       float  End of dc in gap function (also in nm)
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.GapFuncSymmetric(width, thickness, gap, dgap, zmin, zmax)
        return item.sparams(wl)

class sipann_dc_arbitraryantisym(core.ComponentModel):
    """Regression Based form of any directional coupler provided gap function
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    gap: callable,
                    zmin: float,
                    zmax: float,
                    arc_l: float,
                    arc_u: float,
                    width: float=0.5,
                    thickness: float=0.22,
                    sw_angle: float=90,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        width:      float  Width of the waveguide in microns.
        thickness:  float  Thickness of the waveguide in microns.
        gap:        function  Gap between the two waveguides as a function of z (in nm for now)
        zmin:       float  Beginning of dc in gap function (also in nm)
        zmax:       float  End of dc in gap function (also in nm)
        arc_l:      float  Arc length of lower waveguide
        arc_u:      float  Arc length of upper waveguide
        sw_angle    float  Angle in degrees of sidewall of waveguide (between 80 and 90)
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #resize everything to nms
        width     = width*1000
        thickness = thickness*1000

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.GapFuncAntiSymmetric(width, thickness, gap, zmin, zmax, arc_l, arc_u)
        return item.sparams(wl)

@register_component_model
class sipann_dc_crossover1550(core.ComponentModel):
    """Regression Based form of any directional coupler provided gap function
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #load and make gap function
        loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_crossover1550.npz'))
        x = loaded['GAP']
        b = loaded['LENGTH']

        #load scipy.special.binom as a C-compiled function
        addr = get_cython_function_address("scipy.special.cython_special", "binom")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        binom_fn = functype(addr)

        #load all seperate functions that we'll need
        n = len(x) - 1
        @njit
        def binom_in_njit(x, y):
            return binom_fn(x, y)
        @njit
        def bernstein(n,j,t):
            return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
        @njit
        def bez(t):
            n = len(x) - 1
            return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
        @njit
        def dbez(t):
            return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

        #resize everything to nms
        width     = 500
        thickness = 220

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
        return item.sparams(wl)

@register_component_model
class sipann_dc_fifty(core.ComponentModel):
    """Regression Based form of any directional coupler provided gap function
    """
    ports = 4
    cachable = False

    @classmethod
    def s_parameters(cls,
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000):
        """Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        start_freq: float  The starting frequency to obtain s-parameters for.
        stop_freq:  float  The ending frequency to obtain s-parameters for.
        num:        int    The number of points to use between start_freq and stop_freq.
        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`."""
        #load and make gap function
        loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_fifty.npz'))
        x = loaded['GAP']
        b = loaded['LENGTH']

        #load scipy.special.binom as a C-compiled function
        addr = get_cython_function_address("scipy.special.cython_special", "binom")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        binom_fn = functype(addr)

        #load all seperate functions that we'll need
        n = len(x) - 1
        @njit
        def binom_in_njit(x, y):
            return binom_fn(x, y)
        @njit
        def bernstein(n,j,t):
            return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
        @njit
        def bez(t):
            n = len(x) - 1
            return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
        @njit
        def dbez(t):
            return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

        #resize everything to nms
        width     = 500
        thickness = 220

        #switch to wavelength
        c = 299792458
        start_wl = c * 10**9 / stop_freq
        stop_wl  = c * 10**9 / start_freq
        wl       = np.linspace(start_wl, stop_wl, num)

        item = dc.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
        return item.sparams(wl)
