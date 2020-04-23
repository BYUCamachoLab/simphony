import os

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from simphony.elements import Model, interpolate
from simphony.simulation import freq2wl, wl2freq


class ebeam_bdc_te1550(Model):
    """
    A bidirectional coupler optimized for TE polarized light at 1550 nanometers.

    The bidirectional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently 
    interferes lights from two adjacent inputs, efficiently splitting the 
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_bdc_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_bdc_te(Model):
    def __init__(self, thickness, width):
        super().__init__()

class contra_directional_coupler(Model):
    """
    Parameters
    ----------
    w1 : float
        Waveguide width 1 in meters (default 0.45 microns).
    w2 : float
        Waveguide width 2 in meters (default 0.55 microns).
    dW1 : float
        Waveguide 1 corrogation width in meters (default 0.03 microns).
    dW2 : float
        Waveguide 2 corrogation width in meters (default 0.04 microns).
    gap : float
        Waveguide gap in meters (default 0.15 microns).
    p : float
        Grating period in meters (default 0.317 microns).
    N : int
        Number of grating periods (default 300).
    s : bool
        Simulation accuracy (True = high, False = fast).
    a : float
        Gaussian apodization index (default 2.8).
    """
    def __init__(self, w1, w2, dW1, dW2, gap, p, N, s, a):
        l1 = 1500   # starting wavelength
        l2 = 1600   # ending wavelength
        ln = None   # number of sampled points
        super().__init__()

class ebeam_dc_halfring_straight(Model):
    """
    Parameters
    ----------
    gap : float, optional
        Coupling distance between ring and main line in meters (default 0.2 microns).
    radius : float, optional
        Ring radius in meters (default 10 microns).
    width : float, optional
        Waveguide width in meters (default 0.5 microns).
    thickness : float, optional
        Waveguide thickness in meters (default 0.22 microns).
    coupler_length : float, optional
        Length of the coupling edge, squares out ring; in meters (default 0).
    """
    # Consider making a dictionary of filenames so we can filter based on
    # parameters.
    # Perhaps the algorithm would look something like this:
    # Normalize all parameters for their closest matches
    # Compare to all dictionaries to see which have the most matching keys
    # Warn on what's different from designed
    # shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    # print len(shared_items)
    pins = ('n1', 'n2',) #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_halfring_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def __init__(self, gap, radius, width, thickness, couple_length):
        super().__init__()

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_dc_te1550(Model):
    """
    A directional coupler optimized for TE polarized light at 1550 nanometers.

    The directional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently 
    interferes lights from two adjacent inputs, efficiently splitting the 
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    gap : float, optional
        Coupling gap distance, in meters (default 200 microns).
    Lc : float, optional
        Length of coupler, in meters (default 10 microns).
    """
    pins = ('n1', 'n2', 'n3', 'n4')
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def __init__(self, gap=200e-6, Lc=10e-6):
        pass

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_disconnected_te1550(Model):
    pass

class ebeam_disconnected_tm1550(Model):
    pass 

class ebeam_taper_te1550(Model):
    """
    Parameters
    ----------
    w1
    w2
    length
    """
    def __init__(self, w1, w2, length):
        super().__init__()

class ebeam_terminator_te1550(Model):
    """
    A terminator component that dissipates light into free space optimized for
    TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /reference/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1',) #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_terminator_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_terminator_tm1550(Model):
    """
    A terminator component that dissipates light into free space optimized for
    TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /reference/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1',) #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_terminator_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_gc_te1550(Model):
    """
    A grating coupler optimized for TE polarized light at 1550 nanometers.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees 
    [needs citation].

    .. image:: /reference/images/ebeam_gc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        The thickness of the grating coupler, in meters (default 220 
        nanometers). Valid values are 210, 220, or 230 nanometers.
    deltaw : float, optional
        FIXME: unknown parameter (default 0). Valid values are -20, 0, or 20.
    """
    pins = ('n1', 'n2',) #: The default pin names of the device

    def __init__(self, thickness=220e-9, deltaw=0):
        super().__init__()
    
    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])


class ebeam_wg_integral_1550(Model):
    """
    Model for an waveguide optimized for TE polarized light at 1550 nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /reference/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    length : float
        Waveguide length in meters.
    lam0 : float, optional
        Central wavelength for calculation in meters (default 1.55 microns).
    ne : float, optional
        Effective index (default 2.44553).
    ng : float, optional
        Group velocity (default 4.19088).
    nd : float, optional
        Group dispersion (default 3.54275e-04).
    sigma_ne : float, optional
        Standard deviation of the effective index (default 0.05).
    sigma_ng : float, optional
        Standard deviation of the group velocity (default 0.05).
    sigma_nd : float, optional
        Standard deviation of the group dispersion (default 0.0001).

    Notes
    -----
    The `sigma_` values in the parameters are used for monte carlo simulations.
    """
    pins = ('n1', 'n2',) #: The default pin names of the device
    freq_range = (187370000000000.0, 199862000000000.0) #: The valid frequency range for this model.

    def __init__(self, length, lam0=1.55e-06, ne=2.44553, ng=4.19088, nd=0.000354275,
        sigma_ne=0.05, sigma_ng=0.05, sigma_nd=0.0001):
        self.length = length
        self.lam0 = lam0
        self.ne = ne
        self.ng = ng
        self.nd = nd
        self.sigma_ne = sigma_ne
        self.sigma_ng = sigma_ng
        self.sigma_nd = sigma_nd
        self.regenerate_monte_carlo_parameters()
    
    def s_parameters(self, freq):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        start : float
            The starting frequency to obtain s-parameters for (in Hz).
        stop : float
            The ending frequency to obtain s-parameters for (in Hz).
        num : int
            The number of points to use between start_freq and stop_freq.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`, 
            corresponding to the calculated s-parameter matrix, `s`.
        """
        return self.cacl_s_params(freq, self.length, self.lam0, self.ne, self.ng, self.nd)

    def monte_carlo_s_parameters(self, freq):
        """
        Returns a monte carlo (randomized) set of s-parameters.

        In this implementation of the monte carlo routine, random values are
        generated for ne, ng, and nd for each run through of the monte carlo
        simulation. This means that all waveguide elements throughout a single 
        circuit will have the same (random) ne, ng, and nd values. Hence, there
        is correlated randomness in the monte carlo parameters but they are 
        consistent within a single circuit.
        """
        return self.cacl_s_params(freq, self.length, self.lam0, self.rand_ne, self.rand_ng, self.rand_nd)

    def regenerate_monte_carlo_parameters(self):
        self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
        self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
        self.rand_nd = np.random.normal(self.nd, self.sigma_nd)

    @staticmethod
    def cacl_s_params(frequency, length, lam0, ne, ng, nd):
        # Initialize array to hold s-params
        s = np.zeros((len(frequency),2,2), dtype=complex) 

        # Loss calculation
        TE_loss = 700 #dB/m for width 500nm
        alpha = TE_loss/(20*np.log10(np.exp(1)))  

        w = np.asarray(frequency) * 2 * np.pi #get angular frequency from frequency
        w0 = (2*np.pi*SPEED_OF_LIGHT) / lam0 #center frequency (angular)

        #calculation of K
        K = 2*np.pi*ne/lam0 + (ng/SPEED_OF_LIGHT)*(w - w0) - (nd*lam0**2/(4*np.pi*SPEED_OF_LIGHT))*((w - w0)**2)
        
        for x in range(0, len(frequency)): #build s-matrix from K and waveguide length
            s[x,0,1] = s[x,1,0] = np.exp(-alpha*length + (K[x]*length*1j))
        
        return s


class ebeam_y_te1550(Model):
    """
    The y-branch efficiently splits the input between the two outputs.

    .. image:: /reference/images/ebeam_y_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        Waveguide thickness, in meters (default 220 nanometers). Valid values 
        are 210, 220, or 230 nanometers.
    width : float, optional
        Waveguide width, in meters (default 500 nanometers). Valid values are
        480, 500, or 520 nanometers.
    """
    pins = ('n1', 'n2', 'n3') #: The default pin names of the device
    _base_path = 'source_data/y_branch_source'

    def __init__(self, thickness=220e-9, width=500e-9):
        self.thickness = thickness
        self.width = width
        loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_y_1550.npz'))
        self.s_params = (loaded['f'], loaded['s'])
        freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def _get_file(self):
        return 'Ybranch_Thickness ={} width={}.sparam'.format(int(self.thickness*1e9), int(self.width*1e9))

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

class ebeam_y_tm1550(Model):
    """
    The y-branch efficiently splits the input between the two outputs.

    .. image:: /reference/images/ebeam_y_1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1', 'n2', 'n3') #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_y_1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])
