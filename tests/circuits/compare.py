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

lum = parse_lumerical_output('MZIseries1_LUMdata_mag')
sim = load_simphony_output('MZIseries1_SIMdata_mag.npz')
sim = sim['lines'].item()
sim = np.vstack((sim['x_0_to_1'], sim['y_0_to_1'])).T

sim[:,0] = f2w(sim[:,0])
plt.plot()