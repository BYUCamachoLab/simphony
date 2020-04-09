import os

import numpy as np

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

class ebeam_dc_halfring_te1550(Model):
    pins = ('n1', 'n2',) #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_halfring_te1550.npz'))
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
    """
    pins = ('n1', 'n2',) #: The default pin names of the device
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_gc_te1550.npz'))
    s_params = (loaded['f'], loaded['s'])
    freq_range = (s_params[0][0], s_params[0][-1]) #: The valid frequency range for this model.
    
    def s_parameters(self, freq):
        return interpolate(freq, self.s_params[0], self.s_params[1])

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


class ebeam_wg_integral_1550(Model):
    """
    Model for an waveguide optimized for TE polarized light at 1550 nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /reference/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png
    """
    pins = ('n1', 'n2',) #: The default pin names of the device
    freq_range = (187370000000000.0, 199862000000000.0) #: The valid frequency range for this model.

    def __init__(self, length, lam0=1.55e-06, ne=2.44553, ng=4.19088, nd=0.000354275):
        """
        Parameters
        ----------
        length : float
            Waveguide length in meters.
        lam0 : float
            Central wavelength for calculation.
        ne : float
            Effective index.
        ng : float
            Group velocity.
        nd : float
            Group dispersion.
        """
        self.length = length
        self.lam0 = lam0
        self.ne = ne
        self.ng = ng
        self.nd = nd
    
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
        frequency = freq

        # Initialize array to hold s-params
        mat = np.zeros((len(frequency),2,2), dtype=complex) 
        
        c0 = 299792458 #m/s

        # Loss calculation
        TE_loss = 700 #dB/m for width 500nm
        alpha = TE_loss/(20*np.log10(np.exp(1)))  

        w = np.asarray(frequency) * 2 * np.pi #get angular frequency from frequency
        lam0 = self.lam0
        w0 = (2*np.pi*c0) / lam0 #center frequency (angular)
        
        ne = self.ne
        ng = self.ng
        nd = self.nd

        length = self.length

        #calculation of K
        K = 2*np.pi*ne/lam0 + (ng/c0)*(w - w0) - (nd*lam0**2/(4*np.pi*c0))*((w - w0)**2)
        
        for x in range(0, len(frequency)): #build s-matrix from K and waveguide length
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*length + (K[x]*length*1j))
        
        s = mat
        # return (frequency, s)
        return s


class ebeam_y_1550(Model):
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
