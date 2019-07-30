import simphony.core as core
from simphony.core import register_component_model

import os
import numpy as np

@register_component_model
class ebeam_bdc_te1550(core.ComponentModel):
    ports = 4
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_bdc_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True

@register_component_model
class ebeam_dc_halfring_te1550(core.ComponentModel):
    ports = 4
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_halfring_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True

@register_component_model
class ebeam_gc_te1550(core.ComponentModel):
    ports = 2
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_gc_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True

@register_component_model
class ebeam_terminator_te1550(core.ComponentModel):
    ports = 1
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_terminator_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True

@register_component_model
class ebeam_wg_integral_1550(core.ComponentModel):
    """Component model for an ebeam_wg_integral_1550"""
    ports = 2
    cachable = False
    
    @classmethod
    def s_parameters(cls, 
                    length: float=0, 
                    points: list = [],
                    start_freq: float=1.88e+14,
                    stop_freq: float=1.99e+14,
                    num: int=2000,
                    **kwargs):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        length: float   
            Length of the waveguide.
        points: list
            The points that define the waveguide's path.
        start_freq: float
            The starting frequency to obtain s-parameters for.
        stop_freq: float
            The ending frequency to obtain s-parameters for.
        num: int
            The number of points to use between start_freq and stop_freq.
        **kwargs : None
            This is a redundancy in case other parameters are included which
            are unnecessary for calculating the result.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`, 
            corresponding to the calculated s-parameter matrix, `s`.
        """
        frequency = np.linspace(start_freq, stop_freq, num)
        return cls.lumerical_s_params(frequency, length)

    @staticmethod
    def lumerical_s_params(frequency, length: float):
        '''Calculates waveguide s-parameters based on the SiEPIC compact model for waveguides

        Parameters
        ----------
        frequency : np.array
            The frequency array for which to calculate s-parameters.
        length : float   
            Length of the waveguide.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`, 
            corresponding to the calculated s-parameter matrix, `s`.    
        '''
        # Using file that assumes width 500nm and height 220nm
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', "WaveGuideTETMStrip,w=500,h=220.txt")

        # Read info from waveguide s-param file
        with open(filename, 'r') as f:
            coeffs = f.readline().split()
        
        # Initialize array to hold s-params
        mat = np.zeros((len(frequency),2,2), dtype=complex) 
        
        c0 = 299792458 #m/s

        # Loss calculation
        TE_loss = 700 #dB/m for width 500nm
        alpha = TE_loss/(20*np.log10(np.exp(1)))  

        w = np.asarray(frequency) * 2 * np.pi #get angular frequency from frequency
        lam0 = float(coeffs[0]) #center wavelength
        w0 = (2*np.pi*c0) / lam0 #center frequency (angular)
        
        ne = float(coeffs[1]) #effective index
        ng = float(coeffs[3]) #group index
        nd = float(coeffs[5]) #group dispersion
        
        #calculation of K
        K = 2*np.pi*ne/lam0 + (ng/c0)*(w - w0) - (nd*lam0**2/(4*np.pi*c0))*((w - w0)**2)
        
        for x in range(0, len(frequency)): #build s-matrix from K and waveguide length
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*length + (K[x]*length*1j))
        
        s = mat
        return (frequency, s)

@register_component_model
class ebeam_y_1550(core.ComponentModel):
    ports = 3
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_y_1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True
