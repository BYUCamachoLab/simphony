import os

import numpy as np

from simphony.elements import Element
from simphony.simulation import freq2wl, wl2freq

# FIXME: Is interpolating in frequency better than in wavelength?
class ebeam_bdc_te1550(Element):
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_bdc_te1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)
    pins = ('n1', 'n2', 'n3', 'n4')
    ignore = ['loaded'] # optional

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])

class ebeam_bdc_te1550(Element):
    pins = ('n1', 'n2', 'n3', 'n4',)
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_bdc_te1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])

class ebeam_dc_halfring_te1550(Element):
    pins = ('n1', 'n2',)
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_dc_halfring_te1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])

class ebeam_gc_te1550(Element):
    pins = ('n1', 'n2',)
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_gc_te1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name: str = None):
        super().__init__(name=name)
    
    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])

class ebeam_terminator_te1550(Element):
    pins = ('n1',)
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_terminator_te1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])


class ebeam_wg_integral_1550(Element):
    """Component model for an ebeam_wg_integral_1550"""
    pins = ('n1', 'n2',)
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name, length, lam0=1.55e-06, ne=2.44553, ng=4.19088, nd=0.000354275):
        super().__init__(name=name)
        self.length = length
        self.lam0 = lam0
        self.ne = ne
        self.ng = ng
        self.nd = nd
    
    def s_parameters(self, start, stop, num):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        start : float
            The starting frequency to obtain s-parameters for.
        stop : float
            The ending frequency to obtain s-parameters for.
        num : int
            The number of points to use between start_freq and stop_freq.

        Returns
        -------
        (frequency, s) : tuple
            Returns a tuple containing the frequency array, `frequency`, 
            corresponding to the calculated s-parameter matrix, `s`.
        """
        start_freq, stop_freq = wl2freq(start), wl2freq(stop)
        frequency = np.linspace(start_freq, stop_freq, num)

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
        wl = freq2wl(frequency)
        return (wl, s)


class ebeam_y_1550(Element):
    pins = ('n1', 'n2', 'n3')
    loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sparams', 'ebeam_y_1550.npz'))
    s_params = (freq2wl(loaded['f']), loaded['s'])
    wl_bounds = (1.5e-6, 1.6e-6)

    def __init__(self, name: str = None):
        super().__init__(name=name)

    def s_parameters(self, start, stop, num):
        wl = np.linspace(start, stop, num)
        return wl, self.interpolate(wl, self.s_params[0], self.s_params[1])
