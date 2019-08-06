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

plt.figure()

for i in range(1, 5):
    lum = parse_lumerical_output('MZIseries' + str(i) + '_LUMdata_mag')
    sim = load_simphony_output('MZIseries' + str(i) + '_SIMdata_mag.npz')
    sim = sim['lines'].item()
    sim = np.vstack((sim['x_0_to_1'], sim['y_0_to_1'])).T

    sim[:,0] = f2w(sim[:,0]) / 1e3

    plt.plot(lum[:,0], lum[:,1])
    plt.plot(sim[:,0], sim[:,1])
    plt.show()

for i in range(1, 5):
    lum = parse_lumerical_output('MZIseries' + str(i) + '_LUMdata_phase')
    sim = load_simphony_output('MZIseries' + str(i) + '_SIMdata_phase.npz')
    sim = sim['lines'].item()
    sim = np.vstack((sim['x_0_to_1'], sim['y_0_to_1'])).T

    sim[:,0] = f2w(sim[:,0]) / 1e3

    plt.plot(lum[:,0], lum[:,1])
    plt.plot(sim[:,0], sim[:,1])
    plt.plot(lum[:,0], lum[:,1] - sim[:,1])
    plt.show()