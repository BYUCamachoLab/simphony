import os
import numpy as np

from itertools import combinations_with_replacement as comb_w_r

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

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

    INPUT  = cartesian_product([wavelength,width,thickness,angle]) 

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
    coeffs = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'straightCoeffs.npy'))
    return polyCombos@coeffs 


class Model:
    def __init__(self):
        pass

    @staticmethod
    def get_s_params(frequency, length, width, thickness, delta_length):
        '''
        Function that calculates the s-parameters for a waveguide using the ANN model
        Args:
            None
            frequency (frequency array) and length (waveguide length) are used to calculate the s-parameters
        Returns:
            None
            self.s becomes the s-matrix calculated by this function
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
        neff = straightWaveguide(np.transpose(wl), width, thickness, 90)

        #K is calculated from the effective index and wavelength
        K = (2*np.pi*np.true_divide(neff,wl))

        #the s-matrix is built from alpha, K, and the waveguide length
        for x in range(0, len(neff)): 
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*waveguideLength + (K[x]*waveguideLength*1j))
        s = mat
        
        return frequency, s

    @staticmethod
    def about():
        print("ANN model: uses a polynomial fit trained by a neural network to approximate the phase through a waveguide.")