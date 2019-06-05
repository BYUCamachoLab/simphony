"""
simulation.py

Authors: 
    Sequoia Ploeg
    Hyrum Gunther

Dependencies:
- numpy
- copy
- pya
- SiEPIC.ann.models, SiEPIC.ann.netlist
- scipy
- matplotlib
- time

This file contains all classes and functions related to running simulations
of photonic circuits and formatting their data in useful ways.
"""

# import pya
import numpy as np
import copy
from scipy.interpolate import interp1d
import time

from . import models
from . import netlist as nl

class SimulationSetup:
    NUM_INTERP_POINTS = 2000
    INTERP_RANGE = (1.88e+14, 1.99e+14)
    FREQUENCY_RANGE = np.linspace(INTERP_RANGE[0], INTERP_RANGE[1], NUM_INTERP_POINTS) 

    @staticmethod
    def interpolate(freq, sparams):
        func = interp1d(freq, sparams, kind='cubic', axis=0)
        return [SimulationSetup.FREQUENCY_RANGE, func(SimulationSetup.FREQUENCY_RANGE)]




class MathPrefixes:
    TERA = 1e12
    NANO = 1e-9
    c = 299792458




class Simulation:
    def __init__(self, netlist):
        start = time.time()
        self.s_matrix, self.frequency, self.ports, self.external_components = nl.get_sparameters(netlist) 
        self.external_port_list = [-int(x) for x in self.ports]
        self.external_port_list.sort()
        self._rearrangeSMatrix()
        stop = time.time()
        print("Simulation time:", stop-start, "seconds.")
        return

    def _rearrangeSMatrix(self):
        ports = [int(i) for i in self.ports]
        reordered = copy.deepcopy(ports)
        reordered.sort(reverse = True)
        concatenate_order = [ports.index(i) for i in reordered]
        new_s = copy.deepcopy(self.s_matrix)
        reordered_s = np.zeros(self.s_matrix.shape, dtype=complex)

        i = 0
        for idx in concatenate_order:
            reordered_s[:,i,:] = new_s[:,idx,:]
            i += 1
        new_s = copy.deepcopy(reordered_s)
        i = 0
        for idx in concatenate_order:
            reordered_s[:,:,i] = new_s[:,:,idx]
            i += 1
        
        self.s_matrix = copy.deepcopy(reordered_s)

    def frequencyToWavelength(self, frequency):
        return MathPrefixes.c / frequency

    def getMagnitudeByFrequencyTHz(self, fromPort, toPort):
        print("From", fromPort, "to", toPort)
        freq = np.divide(self.frequency, MathPrefixes.TERA)
        mag = abs(self.s_matrix[:,fromPort,toPort])**2
        return freq, mag
    
    def getMagnitudeByWavelengthNm(self, fromPort, toPort):
        wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
        mag = abs(self.s_matrix[:,fromPort,toPort])**2
        return wl, mag

    def getPhaseByFrequencyTHz(self, fromPort, toPort):
        freq = np.divide(self.frequency, MathPrefixes.TERA)
        phase = np.rad2deg(np.unwrap(np.angle(self.s_matrix[:,fromPort,toPort])))
        return freq, phase

    def getPhaseByWavelengthNm(self, fromPort, toPort):
        wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
        phase = np.rad2deg(np.unwrap(np.angle(self.s_matrix[:,fromPort,toPort])))
        print(wl, phase)
        return wl, phase

    def exportSMatrix(self):
        return self.s_matrix, self.frequency

import matplotlib.pyplot as plt
from scipy.io import savemat

class MonteCarloSimulation:
    DEF_NUM_SIMS = 10
    DEF_MU_WIDTH = 0.5
    DEF_SIGMA_WIDTH = 0.005
    DEF_MU_THICKNESS = 0.22
    DEF_SIGMA_THICKNESS = 0.002
    DEF_MU_LENGTH = 0
    DEF_SIGMA_LENGTH = 0
    DEF_DPIN = 1
    DEF_DPOUT = 0
    DEF_SAVEDATA = True
    DEF_DISPTIME = True
    DEF_FILENAME = "monte_carlo.mat"

    def __init__(self, netlist):
        self.netlist = netlist
        # Run simulation with mean width and thickness
        self.mean_s, self.frequency, self.ports, _ = nl.get_sparameters(self.netlist) 

    def monte_carlo_sim(self, 
                        num_sims=DEF_NUM_SIMS, 
                        mu_width=DEF_MU_WIDTH, 
                        sigma_width=DEF_SIGMA_WIDTH, 
                        mu_thickness=DEF_MU_THICKNESS,
                        sigma_thickness=DEF_SIGMA_THICKNESS, 
                        mu_length=DEF_MU_LENGTH, 
                        sigma_length=DEF_SIGMA_LENGTH, 
                        saveData=False, 
                        filename=None, 
                        dispTime=False, 
                        printer=None):
        # Timer
        start = time.time()

        # Random distributions
        random_width = np.random.normal(mu_width, sigma_width, num_sims)
        random_thickness = np.random.normal(mu_thickness, sigma_thickness, num_sims)
        random_deltaLength = np.random.normal(mu_length, sigma_length, num_sims)

        results_shape = np.append(np.asarray([num_sims]), self.mean_s.shape)
        results = np.zeros([dim for dim in results_shape], dtype='complex128')

        # Run simulations with varied width and thickness
        for sim in range(num_sims):
            modified_netlist = copy.deepcopy(self.netlist)
            for component in modified_netlist.component_list:
                if component.__class__.__name__ == "ebeam_wg_integral_1550":
                    component.width = random_width[sim]
                    component.height = random_thickness[sim]
                    # TODO: Implement length monte carlo using random_deltaLength[sim]
            results[sim, :, :, :] = nl.get_sparameters(modified_netlist)[0]
            if ((sim % 10) == 0) and dispTime:
                print(sim)

        # rearrange matrix so matrix indices line up with proper port numbers
        self.ports = [int(i) for i in self.ports]
        rp = copy.deepcopy(self.ports)
        rp.sort(reverse=True)
        concatenate_order = [self.ports.index(i) for i in rp]
        temp_res = copy.deepcopy(results)
        temp_mean = copy.deepcopy(self.mean_s)
        re_res = np.zeros(results_shape, dtype=complex)
        re_mean = np.zeros(self.mean_s.shape, dtype=complex)
        i=0
        for idx in concatenate_order:
            re_res[:,:,i,:]  = temp_res[:,:,idx,:]
            re_mean[:,i,:] = temp_mean[:,idx,:]
            i += 1
        temp_res = copy.deepcopy(re_res)
        temp_mean = copy.deepcopy(re_mean)
        i=0
        for idx in concatenate_order:
            re_res[:,:,:,i] = temp_res[:,:,:,idx]
            re_mean[:,:,i] = temp_mean[:,:,idx]
            i+= 1    
        self.results = copy.deepcopy(re_res)
        self.mean_s = copy.deepcopy(re_mean)

        # print elapsed time if dispTime is True
        stop = time.time()
        if dispTime and printer:
            printer('Total simulation time: ' + str(stop-start) + ' seconds')

        # save MC simulation results to matlab file
        if saveData == True:
            savemat(filename, {'freq':self.frequency, 'results':self.results, 'mean':self.mean_s})

        printer("Simulation complete.")

    def plot(self, num_sims, dpin, dpout):
        plt.figure(1)
        for i in range(num_sims):
            plt.plot(self.frequency, 10*np.log10(abs(self.results[i, :, dpin, dpout])**2), 'b', alpha=0.1)
        plt.plot(self.frequency,  10*np.log10(abs(self.mean_s[:, dpin, dpout])**2), 'k', linewidth=0.5)
        title = 'Monte Carlo Simulation (' + str(num_sims) + ' Runs)'
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.draw()
        plt.show()

    
class MultiInputSimulation(Simulation):
    def __init__(self, netlist):
        super().__init__(netlist)

    def multi_input_simulation(self, inputs: list=[]):
        """
        Parameters
        ----------
        inputs : list
            A 0-indexed list of the ports to be used as inputs.
        """
        active = [0] * len(self.ports)
        for val in inputs:
            active[val] = 1
        self.simulated_matrix = self._measure_s_matrix(active)

    def _measure_s_matrix(self, inputs):
        num_ports = len(inputs)
        inputs = np.array(inputs)
        out = np.zeros([len(self.frequency), num_ports], dtype='complex128')
        for i in range(len(self.frequency)):
            out[i, :] = np.dot(np.reshape(self.s_matrix[i, :, :], [num_ports, num_ports]), inputs.T)
        return out

    def plot(self, output_port):
        plt.figure()
        plt.plot(*self.get_magnitude_by_frequency_thz(output_port))
        plt.title('Multi-Input Simulation')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Gain (dB)')
        plt.draw()
        plt.show()

    def get_magnitude_by_frequency_thz(self, output_port):
        """
        Parameters
        ----------
        output_port : int
            Gets the values at that output port (0-indexed).
        """
        freq = np.divide(self.frequency, MathPrefixes.TERA)
        mag = np.power(np.absolute(self.simulated_matrix[:, output_port]), 2)
        return freq, mag

    def get_magnitude_by_wavelength_nm(self, output_port):
        """
        Parameters
        ----------
        output_port : int
            Gets the values at that output port (0-indexed).
        """
        wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
        mag = np.power(np.absolute(self.simulated_matrix[:, output_port]), 2)
        return wl, mag

    def export_s_matrix(self):
        """Returns the matrix result of the multi-input simulation.

        Returns
        -------
        frequency, matrix: np.array, np.ndarray
        """
        return self.frequency, self.simulated_matrix