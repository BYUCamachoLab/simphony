import os
import numpy as np

class Model:
    def __init__(self):
        pass

    @staticmethod
    def get_s_params(frequency, length, width, thickness, delta_length):
        '''
        Calculates waveguide s-parameters based on the SiEPIC compact model for waveguides
        Args:
            None
            frequency (frequency array) and self.wglen (waveguide length) are used to calculate the s-parameters
        Returns:
            None
            self.s becomes the s-matrix calculated by this function        
        '''
        #using file that assumes width 500nm and height 220nm
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "WaveGuideTETMStrip,w=500,h=220.txt")

        with open(filename, 'r') as f:#read info from waveguide s-param file
            coeffs = f.readline().split()
        
        mat = np.zeros((len(frequency),2,2), dtype=complex) #initialize array to hold s-params
        
        c0 = 299792458 #m/s

        #loss calculation
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
        return frequency, s

    @staticmethod
    def about():
        print("Lumerical model: uses a third order polynomial to approximate the phase through a waveguide.")