import simphony.core as core
from simphony.core import register_component_model

import os
import numpy as np
from itertools import combinations_with_replacement as comb_w_r

@register_component_model
class ann_wg_integral(core.ComponentModel):
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
