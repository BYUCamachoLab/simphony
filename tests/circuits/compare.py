import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def parse_lumerical_output(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=3)

def load_simphony_output(filename):
    return np.load(filename, allow_pickle=True)

def f2w(frequency):
    '''Converts from frequency to wavelength.'''
    return c / frequency

def w2f(wavelength):
    '''Converts from wavelength to frequency.'''
    return c / wavelength

def find_nearest(array, value):
    array = np.asarray(array)
    if not isinstance(value, (list, tuple, np.ndarray)):
        idx = (np.abs(array - value)).argmin()
    else:
        idx = [(np.abs(array - i)).argmin() for i in value]
    # return array[idx]
    return idx

plt.figure()

def compare_magnitudes():
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        lum = parse_lumerical_output('MZIseries' + str(i) + '_LUMdata_mag')
        sim = load_simphony_output('MZIseries' + str(i) + '_SIMdata_mag.npz')
        sim = sim['lines'].item()
        sim = np.vstack((sim['x_0_to_1'], sim['y_0_to_1'])).T

        sim[:,0] = f2w(sim[:,0]) / 1e3

        plt.plot(lum[:,0], lum[:,1])
        plt.plot(sim[:,0], sim[:,1])
        plt.title("MZIseries" + str(i))
    plt.tight_layout()
    plt.show()

def compare_phases():
    for i in range(1, 5):
        plt.subplot(2,2,i)
        lum = parse_lumerical_output('MZIseries' + str(i) + '_LUMdata_phase')
        sim = load_simphony_output('MZIseries' + str(i) + '_SIMdata_phase.npz')
        sim = sim['lines'].item()
        sim = np.vstack((sim['x_0_to_1'], sim['y_0_to_1'])).T

        sim[:,0] = f2w(sim[:,0]) / 1e3

        plt.plot(lum[:,0], lum[:,1])
        plt.plot(sim[:,0], sim[:,1])

        lin = np.linspace(1520, 1580, num=150)
        lum_idxs = find_nearest(lum[:,0], lin)
        sim_idxs = find_nearest(sim[:,0], lin)
        
        lum = lum[lum_idxs, :]
        sim = sim[sim_idxs, :]
        plt.plot(lum[:,0], lum[:,1] - sim[:,1])
        plt.title("MZIseries" + str(i))
    plt.tight_layout()
    plt.show()

compare_magnitudes()
compare_phases()