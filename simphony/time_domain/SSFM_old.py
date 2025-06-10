import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib import cm

def getFreqRangeFromTime(time):
    return fftshift(fftfreq(len(time), d=time[1]-time[0]))

def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) #Get phase starting from 1st entry
    phi=phi-phi[int(len(phi)/2)]   #Center phase on middle entry
    return phi


def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0]  - (phi[1]  - phi[0]  ),axis=0) #Change in phase. Prepend to ensure consistent array size
    dt  =np.diff(time,prepend = time[0] - (time[1] - time[0] ),axis=0) #Change in time.  Prepend to ensure consistent array size

    return -1.0/(2*np.pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt

class SIM_config:
    def __init__(self,N,dt):
        self.number_of_points=N
        self.time_step=dt
        t=np.linspace(0,N*dt,N)
        self.t=t-np.mean(t)
        self.tmin=self.t[0]
        self.tmax=self.t[-1]

        self.f=getFreqRangeFromTime(self.t)
        self.fmin=self.f[0]
        self.fmax=self.f[-1]
        self.freq_step=self.f[1]-self.f[0]

        self.describe_config()

    def describe_config(self):
        print("### Configuration Parameters ###")
        print(f" Number of points = {self.number_of_points}")
        print(f" Start time, tmin = {self.tmin*1e12}ps")
        print(f" Stop time, tmax = {self.tmax*1e12}ps")
        print(f" Time resolution, dt = {self.time_step*1e12}ps")
        print("  ")
        print(f" Start frequency = {self.fmin/1e12}THz")
        print(f" Stop frequency = {self.fmax/1e12}THz")
        print(f" Frequency resolution = {self.freq_step/1e6}MHz")
        print( "   ")

#Function returns pulse power or spectrum PSD
def getPower(amplitude):
    return np.abs(amplitude)**2

#Function gets the energy of a pulse pulse or spectrum by integrating the power
def getEnergy(time_or_frequency,amplitude):
    return np.trapz(getPower(amplitude),time_or_frequency)

#TODO: Add support for different carrier frequencies. Hint: Multiply by complex exponential!
#TODO: Add support for pre-chirped pulses.
def GaussianPulse(time,amplitude,duration,offset,chirp,order):
  assert 1 <= order, f"Error: Order of gaussian pulse is {order}. Must be >=1"
  return amplitude*np.exp(- (1+1j*chirp)/2*((time-offset)/(duration))**(2*np.floor(order)))*(1+0j)

def getSpectrumFromPulse(time,pulse_amplitude):
    pulseEnergy=getEnergy(time,pulse_amplitude) #Get pulse energy
    f=getFreqRangeFromTime(time)
    dt=time[1]-time[0]

    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt #Take FFT and do shift
    spectrumEnergy=getEnergy(f, spectrum_amplitude) #Get spectrum energy

    err=np.abs((pulseEnergy/spectrumEnergy-1))

    # assert( err<1e-7 ), f'ERROR = {err}: Energy changed when going from Pulse to Spectrum!!!'

    return spectrum_amplitude



#Equivalent function for getting time base from frequency range
def getTimeFromFrequency(frequency):
    return fftshift(fftfreq(len(frequency), d=frequency[1]-frequency[0]))


#Equivalent function for getting pulse from spectrum
def getPulseFromSpectrum(frequency,spectrum_amplitude):

    spectrumEnergy=getEnergy(frequency, spectrum_amplitude)

    time = getTimeFromFrequency(frequency)
    dt = time[1]-time[0]

    pulse = ifft(ifftshift(spectrum_amplitude))/dt
    pulseEnergy = getEnergy(time, pulse)

    err=np.abs((pulseEnergy/spectrumEnergy-1))

    # assert( err<1e-7   ), f'ERROR = {err}: Energy changed when going from Spectrum to Pulse!!!'

    return pulse

#Equivalent function for generating a Gaussian spectrum
def GaussianSpectrum(frequency,amplitude,bandwidth):
    time = getTimeFromFrequency(frequency)
    return getSpectrumFromPulse(time, GaussianPulse(time, amplitude, 1/bandwidth, 0,0,1))

#Class for holding info about the fiber
class Fiber_config:
  def __init__(self,nsteps,L,gamma,beta2,alpha_dB_per_m):
      self.nsteps=nsteps
      self.ntraces = self.nsteps+1 #Note: If we want to do 100 steps, we will get 101 calculated pulses (zeroth at the input + 100 computed ones)
      self.Length=L
      self.dz=L/nsteps
      self.zlocs=np.linspace(0,L,self.ntraces) #Locations of each calculated pulse
      self.gamma=gamma
      self.beta2=beta2
      self.alpha_dB_per_m=alpha_dB_per_m
      self.alpha_Np_per_m = self.alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations
      #TODO: Make alpha frequency dependent.

def SSFM(fiber:Fiber_config,sim:SIM_config, pulse):

    #Initialize arrays to store pulse and spectrum throughout fiber
    pulseMatrix = np.zeros((fiber.nsteps+1,sim.number_of_points ) )*(1+0j)
    spectrumMatrix = np.copy(pulseMatrix)
    pulseMatrix[0,:]=pulse
    spectrumMatrix[0,:] = getSpectrumFromPulse(sim.t, pulse)

    #Pre-calculate effect of dispersion and loss as it's the same everywhere
    disp_and_loss=np.exp((1j*fiber.beta2/2*(2*np.pi*sim.f)**2-fiber.alpha_Np_per_m/2)*fiber.dz )

    #Precalculate constants for nonlinearity
    nonlinearity=1j*fiber.gamma*fiber.dz

    for n in range(fiber.nsteps):
        pulse*=np.exp(nonlinearity*getPower(pulse)) # Apply nonlinearity
        spectrum = getSpectrumFromPulse(sim.t, pulse)*disp_and_loss # Go to spectral domain and apply disp and loss
        pulse=getPulseFromSpectrum(sim.f, spectrum) # Return to time domain

        # Store results and repeat
        pulseMatrix[n+1,:]=pulse
        spectrumMatrix[n+1,:]=spectrum

    #Return results
    return pulseMatrix, spectrumMatrix


N  = 2**15 #Number of points
N  = 300000 #Number of points
dt = 0.0001e-12 #Time resolution [s]
t=np.linspace(0,N*dt,N) #Time step array
t=t-np.mean(t)          #Center so middle entry is t=0
#Initialize class
sim_config=SIM_config(N,dt)



#Define fiberulation parameters
Length          = 5
nsteps          = 2**8

gamma           = 10e-3
beta2           = 100e3
beta2          *= (1e-30)
alpha_dB_per_m  = 0.2e-3

#  Initialize class
fiber=Fiber_config(nsteps, Length, gamma, beta2, alpha_dB_per_m)


#Initialize Gaussian pulse
amplitude = 1                       #Amplitude in units of sqrt(W)
duration  = 0.5e-12   #Pulse 1/e^2 duration [s]
offset    = 0                #Time offset
testChirp = 0
testOrder = 1
testCarrierFreq=0
testPulse=GaussianPulse(t, amplitude, duration, offset, testChirp, testOrder)
plt.plot(t, testPulse)
plt.show()

pulseMatrix, spectrumMatrix = SSFM(fiber,sim_config,testPulse)

plt.plot(t, np.abs(pulseMatrix[0,:])**2)
plt.plot(t, np.abs(pulseMatrix[-1,:])**2)
plt.show()
pass