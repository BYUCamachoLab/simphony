

import simphony.core as core

import os
import numpy as np
import cmath as cm
import copy
from itertools import combinations_with_replacement as comb_w_r

class ebeam_bdc_te1550(core.ComponentModel):

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=self._read_s_parameters(), cachable=True)

    @staticmethod
    def _read_s_parameters():
        """Returns the s-parameters across some frequency range for the ebeam_bdc_te1550 model
        in the form [frequency, s-parameters].
        """
        numports = 4
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_bdc_te1550", "EBeam_1550_TE_BDC.sparam")
        F = []
        S = []
        with open(filename, "r") as fid:
            line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(','))[0])
            S = np.zeros((numrows, numports, numports), dtype='complex128')
            r = m = n = 0
            for line in fid:
                if(line[0] == '('):
                    continue
                data = line.split()
                data = list(map(float, data))
                if(m == 0 and n == 0):
                    F.append(data[0])
                S[r,m,n] = data[1] * np.exp(1j*data[2])
                r += 1
                if(r == numrows):
                    r = 0
                    m += 1
                    if(m == numports):
                        m = 0
                        n += 1
                        if(n == numports):
                            break
        return (F, S)



class ebeam_dc_halfring_te1550(core.ComponentModel):

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=self._read_s_parameters(), cachable=True)
        
    @staticmethod
    def _read_s_parameters():
        """Returns the s-parameters across some frequency range for the ebeam_dc_halfring_te1550 model
        in the form [frequency, s-parameters].

        Parameters
        ----------
        numports : int
            The number of ports the photonic component has.
        """
        numports = 4
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_dc_halfring_te1550", "te_ebeam_dc_halfring_straight_gap=30nm_radius=3um_width=520nm_thickness=210nm_CoupleLength=0um.dat")
        F = []
        S = []
        with open(filename, "r") as fid:
            for i in range(5):
                line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(','))[0])
            S = np.zeros((numrows, numports, numports), dtype='complex128')
            r = m = n = 0
            for line in fid:
                if(line[0] == '('):
                    continue
                data = line.split()
                data = list(map(float, data))
                if(m == 0 and n == 0):
                    F.append(data[0])
                S[r,m,n] = data[1] * np.exp(1j*data[2])
                r += 1
                if(r == numrows):
                    r = 0
                    m += 1
                    if(m == numports):
                        m = 0
                        n += 1
                        if(n == numports):
                            break
        return (F, S)



class ebeam_gc_te1550(core.ComponentModel):

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=self._read_s_parameters(), cachable=True)
        
    @staticmethod
    def _read_s_parameters():
        """Returns the s-parameters across some frequency range for the ebeam_y_1550 model
        in the form [frequency, s-parameters].

        Parameters
        ----------
        numports : int
            The number of ports the photonic component has.
        """
        numports = 2
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_gc_te1550", "GC_TE1550_thickness=220 deltaw=0.txt")
        with open(filename) as fid:
            #grating coupler compact models have 100 points for each s-matrix index
            arrlen = 100
            
            lines = fid.readlines()
            F = np.zeros(arrlen)
            S = np.zeros((arrlen,2,2), 'complex128')
            for i in range(0, arrlen):
                words = lines[i].split()
                F[i] = float(words[0])
                S[i,0,0] = cm.rect(float(words[1]), float(words[2]))
                S[i,0,1] = cm.rect(float(words[3]), float(words[4]))
                S[i,1,0] = cm.rect(float(words[5]), float(words[6]))
                S[i,1,1] = cm.rect(float(words[7]), float(words[8]))
            F = F[::-1]
            S = S[::-1,:,:]
        return (F, S)


class ebeam_terminator_te1550(core.ComponentModel):

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=self._read_s_parameters(), cachable=True)
        
    @staticmethod
    def _read_s_parameters():
        """Returns the s-parameters across some frequency range for the ebeam_terminator_te1550 model
        in the form [frequency, s-parameters].

        Parameters
        ----------
        numports : int
            The number of ports the photonic component has.
        """
        numports = 1
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_terminator_te1550", "nanotaper_w1=500,w2=60,L=10_TE.sparam")
        F = []
        S = []
        with open(filename, "r") as fid:
            line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(','))[0])
            S = np.zeros((numrows, numports, numports), dtype='complex128')
            r = m = n = 0
            for line in fid:
                if(line[0] == '('):
                    continue
                data = line.split()
                data = list(map(float, data))
                if(m == 0 and n == 0):
                    F.append(data[0])
                S[r,m,n] = data[1] * np.exp(1j*data[2])
                r += 1
                if(r == numrows):
                    r = 0
                    m += 1
                    if(m == numports):
                        m = 0
                        n += 1
                        if(n == numports):
                            break
        return (F, S)



class ebeam_wg_integral_1550(core.ComponentModel):
    """Component model for an ebeam_wg_integral_1550"""

    MODELS = {
        "Artificial Neural Network": 'ann_s_params',
        "Lumerical": 'lumerical_s_params',
    }

    OPTIONS = {
        'model': "Artificial Neural Network",
        'frequency': np.linspace(1.88e+14, 1.99e+14, num=2000),
        'length': 0,
        'width': 0.5,
        'thickness': 0.22,
        'radius': 0,
        'points': [],
        'delta_length': 0,
    }

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=None, cachable=False)
    
    @classmethod
    def get_s_parameters(cls, extras: dict={}):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        extras : dict
            Takes a dictionary containing the following (optional) values:
            model: string   Representation of the model.
                options: "Artificial Neural Network" (default) | "Lumerical"
            length: float   Length of the waveguide.
                default: 0.0
            width: float    Width of the waveguide in microns. 
                default: 0.5
            thickness: float   Thickness of the waveguide in microns.
                default: 0.22
            radius: float   Bend radius of bends in the waveguide.
                default: 0
            delta_length: float     Only used in monte carlo simulations.
                default: 0
        """
        options = copy.deepcopy(cls.OPTIONS)
        for key, val in extras.items():
            options[key] = val
        model = getattr(cls, cls.MODELS[options['model']])
        return model(**options)

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

        INPUT  = ebeam_wg_integral_1550.cartesian_product([wavelength,width,thickness,angle]) 

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
    def ann_s_params(frequency, length, width, thickness, delta_length, **kwargs):
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
        neff = ebeam_wg_integral_1550.straightWaveguide(np.transpose(wl), width, thickness, 90)

        #K is calculated from the effective index and wavelength
        K = (2*np.pi*np.true_divide(neff,wl))

        #the s-matrix is built from alpha, K, and the waveguide length
        for x in range(0, len(neff)): 
            mat[x,0,1] = mat[x,1,0] = np.exp(-alpha*waveguideLength + (K[x]*waveguideLength*1j))
        s = mat
        
        return (frequency, s)

    @staticmethod
    def lumerical_s_params(frequency, length, width, thickness, delta_length, **kwargs):
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

    def __init__(self):
        super().__init__(component_type=type(self).__name__, s_parameters=self._read_s_parameters(), cachable=True)
        
    @staticmethod
    def _read_s_parameters():
        """Returns the s-parameters across some frequency range for the ebeam_y_1550 model
        in the form [frequency, s-parameters].

        Parameters
        ----------
        numports : int
            The number of ports the photonic component has.
        """
        numports = 3
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sparams", "ebeam_y_1550", "Ybranch_Thickness =220 width=500.sparam")
        F = []
        S = []
        with open(filename, "r") as fid:
            line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(','))[0])
            S = np.zeros((numrows, numports, numports), dtype='complex128')
            r = m = n = 0
            for line in fid:
                if(line[0] == '('):
                    continue
                data = line.split()
                data = list(map(float, data))
                if(m == 0 and n == 0):
                    F.append(data[0])
                S[r,m,n] = data[1] * np.exp(1j*data[2])
                r += 1
                if(r == numrows):
                    r = 0
                    m += 1
                    if(m == numports):
                        m = 0
                        n += 1
                        if(n == numports):
                            break
        return (F, S)