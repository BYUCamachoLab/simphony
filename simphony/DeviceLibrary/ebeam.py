import simphony.core as core

import os
import numpy as np

class ebeam_bdc_te1550(core.ComponentModel):
    ports = 4
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_bdc_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True


class ebeam_dc_halfring_te1550(core.ComponentModel):
    ports = 4
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_halfring_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True


class ebeam_gc_te1550(core.ComponentModel):
    ports = 2
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_gc_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True


class ebeam_terminator_te1550(core.ComponentModel):
    ports = 1
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_terminator_te1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True


class ebeam_wg_integral_1550(core.ComponentModel):
    """Component model for an ebeam_wg_integral_1550"""
    ports = 2
    cachable = False
    
    @classmethod
    def s_parameters(cls, 
                    length=0, 
                    points = [],
                    start_freq=1.88e+14,
                    stop_freq=1.99e+14,
                    num=2000,
                    **kwargs):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        length: float   
            Length of the waveguide.
        points: list
        start_freq: float
        stop_freq: float
        num: int
            The number of points to use between start_freq and stop_freq.
        """
        frequency = np.linspace(start_freq, stop_freq, num)
        return cls.lumerical_s_params(frequency, length)

    @staticmethod
    def lumerical_s_params(frequency, length):
        '''
        Calculates waveguide s-parameters based on the SiEPIC compact model for waveguides
        Args:
            None
            frequency (frequency array) and self.wglen (waveguide length) are used to calculate the s-parameters
        Returns:
            None
            self.s becomes the s-matrix calculated by this function        
        '''
        # Using file that assumes width 500nm and height 220nm
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_wg_integral_1550', "WaveGuideTETMStrip,w=500,h=220.txt")

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


class ebeam_y_1550(core.ComponentModel):
    ports = 3
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_y_1550.npz'))
    s_parameters = (loaded['f'], loaded['s'])
    cachable = True
