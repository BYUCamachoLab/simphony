# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.library.sipann
=======================

This package contains the models for the SiPANN integration.
"""

import os
from itertools import combinations_with_replacement as comb_w_r

# import ctypes
from numba import njit
from numba.extending import get_cython_function_address
import ctypes
import numpy as np
from scipy import special
from SiPANN import scee

from simphony.elements import Model, interpolate
from simphony.simulation import freq2wl, wl2freq


class sipann_wg_integral(Model):
    """Neural-net trained model of a waveguide.

    A waveguide easily connects other components within the circuit.
    The SiP-ANN waveguide is different from the EBeam package since its
    values are calculated based on a regression fit to simulation data.

    .. image:: /reference/images/ebeam_wg_integral_1550.png
        :alt: ebeam_wg_integral_1550.png

    Parameters
    ----------
    length : float
        The length of the waveguide in microns.
    width : float
        The width of the waveguide in microns.
    thickness : float
        The thickness of the waveguide in microns.
    radius : float
        The radius of the waveguide bends in microns.
    """
    pins = ('n1', 'n2') #: The default pin names of the device
    freq_range = (187370000000000.0, 199862000000000.0) #: The valid frequency range for this model.

    def __init__(self, length, width=0.5, thickness=0.22, radius=5, 
        sigma_length=0.0, sigma_width=0.005, sigma_thickness=0.002):
        self.length = length
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.sigma_length = sigma_length
        self.sigma_width = sigma_width
        self.sigma_thickness = sigma_thickness
        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq):
        s = self.ann_s_params(freq, self.length, self.width, self.thickness)
        return s

    def monte_carlo_s_parameters(self, freq, *args, **kwargs):
        return self.ann_s_params(freq, self.rand_length, self.rand_width, self.rand_thickness)

    def regenerate_monte_carlo_parameters(self):
        self.rand_width = np.random.normal(self.width, self.sigma_width)
        self.rand_thickness = np.random.normal(self.thickness, self.sigma_thickness)
        self.rand_length = np.random.normal(self.length, self.sigma_length)

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

        INPUT  = sipann_wg_integral.cartesian_product([wavelength,width,thickness,angle])

        #Get all possible combinations to use
        degree = 4
        features = 4
        combos = []
        for i in range(degree+1):
            combos += [k for k in comb_w_r(range(features),i)]

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
    def ann_s_params(frequency, length, width, thickness):
        '''
        Function that calculates the s-parameters for a waveguide using the ANN model

        Parameters
        ----------
        frequency : np.array
        length : float
        width : float
        thickness : float

        Returns
        -------
        s : np.ndarray
            Returns a tuple containing the frequency array, `frequency`,
            corresponding to the calculated s-parameter matrix, `s`.
        '''

        mat = np.zeros((len(frequency),2,2), dtype=complex)

        c0 = 299792458 #m/s
        mode = 0 #TE
        TE_loss = 700 #dB/m for width 500nm
        alpha = TE_loss/(20*np.log10(np.exp(1))) #assuming lossless waveguide
        waveguideLength = length

        #calculate wavelength
        wl = np.true_divide(c0,frequency)

        # effective index is calculated by the ANN
        neff = sipann_wg_integral.straightWaveguide(np.transpose(wl), width, thickness, 90)

        #K is calculated from the effective index and wavelength
        K = (2*np.pi*np.true_divide(neff,wl))

        #the s-matrix is built from alpha, K, and the waveguide length
        for x in range(0, len(neff)):
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*waveguideLength + (K[x]*waveguideLength*1j))
        s = mat

        return s


class sipann_scee_straight(Model):
    """Regression Based Closed Form solution of parallel straight waveguides

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float
        Distance between the two waveguides edge in microns.
    length : float
        Length of both waveguides in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, length=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of parameterized parallel waveguides.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.StraightCoupler(width=self.width, thickness=self.thickness, gap=self.gap, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_halfracetrack(Model):
    """Regression Based Closed Form solution of half of a racetrack ring resonator

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    radius : float
        Distance from center of ring to middle of waveguide in microns.
    gap : float
        Minimum distance from ring waveguide edge to straight waveguide edge in microns.
    length : float
        Length of straight portion of ring waveguide in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, length=2.5, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized half racetrack ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.HalfRacetrack(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_halfring(Model):
    """Regression Based Closed Form solution of half of a ring resonator

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    radius : float  
        Radius of bent portions of waveguide
    sw_angle : float  
        Angle in degrees of sidewall of waveguide (between 80 and 90)
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized half ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.HalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_standard(Model):
    """Regression Based Closed Form solution of a standard shaped directional coupler

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness : float
        Thickness of waveguide in microns
    gap : float
        Minimum distance between the two waveguides edge in microns.
    length : float
        Length of the straight portion of both waveguides in microns.
    H : float
        Horizontal distance between end of coupler until straight portion in microns.
    H : float
        Vertical distance between end of coupler until straight portion in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, length=5.0, H=2.0, V=2.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.length    = length*1000
        self.H         = H*1000
        self.V         = V*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized standard directional coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.Standard(width=self.width, thickness=self.thickness, gap=self.gap, length=self.length, H=self.H, V=self.V, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_doublehalfring(Model):
    """Regression Based Closed Form solution of 2 coupling half rings

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    gap : float
            Minimum distance from ring waveguide edge to other ring waveguide edge in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of parameterized 2 coupling half rings.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.DoubleHalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_angledhalfring(Model):
    """Regression Based Closed Form solution of half of a ring resonator pushed into a 
    straight coupling waveguide.

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    radius : float  
        Radius of bent portions of waveguide
    theta : float
        Angle that the straight waveguide is curved in radians (???).
    sw_angle : float  
        Angle in degrees of sidewall of waveguide (between 80 and 90)
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, theta=np.pi/4, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.theta     = theta
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized angled half ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.AngledHalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, theta=self.theta, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_arbitrary_antisym(Model):
    """Regression Based Solution for arbitrarily shaped anti-symmetric coupler

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : function
        Gap function as one progresses along the waveguide (nm)
    zmin : float
        Where to begin integration in the gap function (nm)
    zmax : float
        Where to end integration in the gap function (nm)
    arc1, arc2, arc3, arc4 : float
        Arclength from entrance of each port till minimum coupling point (nm)
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, gap, zmin, zmax, arc1, arc2, arc3, arc4, width=0.5, thickness=0.22, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap
        self.zmin      = zmin
        self.zmax      = zmax
        self.arc1      = arc1
        self.arc2      = arc2
        self.arc3      = arc3
        self.arc4      = arc4
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized arbitrarily shaped anti-symmetric coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncAntiSymmetric(width=self.width, thickness=self.thickness, gap=self.gap, zmin=self.zmin, zmax=self.zmax, arc1=self.arc1, arc2=self.arc2, arc3=self.arc3, arc4=self.arc4, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_arbitrary_sym(Model):
    """Regression Based Solution for arbitrarily shaped symmetric coupler

    # .. comment image:: /reference/images/ebeam_bdc_te1550.png
    #     :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : function
        Gap function as one progresses along the waveguide (nm)
    dgap : function
        Derivative of the gap function (nm)
    zmin : float
        Where to begin integration in the gap function (nm)
    zmax : float
        Where to end integration in the gap function (nm)
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, gap, dgap, zmin, zmax, width=0.5, thickness=0.22, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap
        self.dgap      = dgap
        self.zmin      = zmin
        self.zmax      = zmax
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized arbitrarily shaped symmetric coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncSymmetric(width=self.width, thickness=self.thickness, gap=self.gap, dgap=self.gap, zmin=self.zmin, zmax=self.zmax, sw_angle=self.sw_angle)
        return item.sparams(wl)


class sipann_scee_crossover1550(Model):
    """Regression Based form of a crossover at lambda=1550nm

    Regression based form of a 100/0 directional coupler.

    .. image:: /reference/images/sipann_scee_crossover1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_scee_crossover1550_s.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

    ############## Use the below function to manually run computation of crossover ################
    ############## Can be used to expand range of sparams saved before as needed   ################
    # def s_parameters(self, freq):
    #     """Get the s-parameters of a parameterized 50/50 directional coupler.
    #     Parameters
    #     ----------
    #     freq : np.ndarray
    #         A frequency array to calculate s-parameters over (in Hz).

    #     Returns
    #     -------
    #     s : np.ndarray
    #         Returns the calculated s-parameter matrix.
    #     """
    #     #load and make gap function
    #     loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_crossover1550.npz'))
    #     x = loaded['GAP']
    #     b = loaded['LENGTH']

    #     #load scipy.special.binom as a C-compiled function
    #     addr = get_cython_function_address("scipy.special.cython_special", "binom")
    #     functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
    #     binom_fn = functype(addr)

    #     #load all seperate functions that we'll need
    #     n = len(x) - 1
    #     @njit
    #     def binom_in_njit(x, y):
    #         return binom_fn(x, y)
    #     @njit
    #     def bernstein(n,j,t):
    #         return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
    #     @njit
    #     def bez(t):
    #         n = len(x) - 1
    #         return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
    #     @njit
    #     def dbez(t):
    #         return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

    #     #resize everything to nms
    #     width     = 500
    #     thickness = 220

    #     #switch to wavelength
    #     wl = freq2wl(freq) * 1e9

    #     item = scee.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
    #     return item.sparams(wl)


class sipann_scee_fifty(Model):
    """Regression Based form of a 50/50 directional coupler at lambda=1550nm
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_scee_fifty_s.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])
    
    ############## Use the below function to manually run computation of 50/50   ################
    ############## Can be used to expand range of sparams saved before as needed ################
    # def s_parameters(self, freq):
    #     """Get the s-parameters of a parameterized 50/50 directional coupler.
    #     Parameters
    #     ----------
    #     freq : np.ndarray
    #         A frequency array to calculate s-parameters over (in Hz).

    #     Returns
    #     -------
    #     s : np.ndarray
    #         Returns the calculated s-parameter matrix.
    #     """
    #     #load and make gap function
    #     loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'sipann_fifty.npz'))
    #     x = loaded['GAP']
    #     b = loaded['LENGTH']

    #     #load scipy.special.binom as a C-compiled function
    #     addr = get_cython_function_address("scipy.special.cython_special", "binom")
    #     functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
    #     binom_fn = functype(addr)

    #     #load all seperate functions that we'll need
    #     n = len(x) - 1
    #     @njit
    #     def binom_in_njit(x, y):
    #         return binom_fn(x, y)
    #     @njit
    #     def bernstein(n,j,t):
    #         return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
    #     @njit
    #     def bez(t):
    #         n = len(x) - 1
    #         return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
    #     @njit
    #     def dbez(t):
    #         return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

    #     #resize everything to nms
    #     width     = 500
    #     thickness = 220

    #     #switch to wavelength
    #     wl = freq2wl(freq) * 1e9

    #     item = scee.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
    #     return item.sparams(wl)


class sipann_scee_waveguide(Model):
    """Lossless model for a straight waveguide. 
    
    Simple model that makes sparameters for a straight waveguide. May not be 
    the best option, but plays nice with other models in SCEE. Ports are numbered as::

        |  1 ----------- 2   |

    Parameters
    ----------
    width : float
        Width of the waveguide in microns
    thickness : float
        Thickness of waveguide in microns
    length : float
        Length of waveguide in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, length=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized waveguide.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.Waveguide(width=self.width, thickness=self.thickness, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)