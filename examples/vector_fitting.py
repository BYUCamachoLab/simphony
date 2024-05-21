import numpy as np
import matplotlib.pyplot as plt
from simphony.fast_vector_fitting import FastVF, VF_Options

# Patient 1 from 
# M. K. Sharp, G. M. Pantalos, L. Minich, L. Y. Tani, E. C. McGough, and
# J. A. Hawkins. Aortic input impedance in infants and children. Journal of
# Applied Physiology, 88(6):2227-2239, 2000.

# Magnitude and phase
Zmag = np.array([3125.90, 448.66, 340.70, 492.55, 450.52, 906.79, 574.12, 456.40, 570.80, 546.01, 434.76])
Zphase = np.pi / 180 * np.array([0.00, -25.71, 5.64, 23.14, 33.82, -6.50, 27.56, 14.03, 16.24, 34.98, 25.55])

# Impedance [dyn*s*cm^-5]
Z = Zmag * np.exp(1j * Zphase)

# Heart rate [1/minute]
HR = 152.4

# Period [s]
T = 60 / HR

omega0 = 2 * np.pi / T

# Frequency response
omega = np.arange(11) * omega0
Z = Z.reshape(len(Z), 1, 1)

def ComputeModelResponse(omega, R0, Rr, Rc, pr, pc):
    """
    Compute the frequency response of a Vector Fitting model

    Parameters:
    - omega: frequency samples, column vector. This is angular frequency (omega = 2*pi*f)
    - R0: constant coefficient
    - Rr: residues of real poles, 3D array. First dimension corresponds to system outputs. 
          Second dimension to system inputs. Third dimension corresponds to the various poles.
    - Rc: residues of complex conjugate pole pairs (only one per pair)
    - pr: real poles, column vector
    - pc: complex poles, column vector. Only one per pair of conjugate poles

    Returns:
    - H: model response samples, 3D array. First dimension corresponds to system outputs.
         Second dimension to system inputs. Third dimension corresponds to frequency.
    """
    qbar = R0.shape[0]     # number of outputs
    mbar = R0.shape[1]     # number of inputs
    kbar = len(omega)      # number of frequency points

    nr = len(pr)           # number of real poles
    nc = len(pc)           # number of complex conjugate pairs

    # Preallocate space for H
    H = np.zeros((qbar, mbar, kbar), dtype=complex)

    # Compute the model response
    for ik in range(kbar):
        H[:, :, ik] = R0
        for ir in range(nr):
            H[:, :, ik] += Rr[:, :, ir] / (1j * omega[ik] - pr[ir])
        for ic in range(nc):
            H[:, :, ik] += Rc[:, :, ic] / (1j * omega[ik] - pc[ic])
            H[:, :, ik] += np.conj(Rc[:, :, ic]) / (1j * omega[ik] - np.conj(pc[ic]))

    return H

# Fitting
Order = 10
# Because of the noise and poor sampling in the original samples, we need
# to essentially exclude the first convergence test (never satisfied)
options = VF_Options()
options.poles_estimation_threshold = 100

for Order in range(1, 10):
    model = FastVF(omega, Z, Order, options)

    # Compute model response over a finer grid
    omega_model = np.linspace(min(omega), max(omega), 100)

    Z_model = ComputeModelResponse(omega_model, model.R0, model.Rr, model.Rc, model.poles_real, model.poles_complex)

    # Plot the frequency response of the model
    plt.plot(omega_model / (2 * np.pi), np.abs(np.squeeze(Z_model)), 'r-.', linewidth=1.5)
    plt.plot(omega / (2 * np.pi), np.abs(np.squeeze(Z)), 'bo')
    plt.legend([f'Model (order={Order})', 'Samples' ])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dyn·s·cm⁻⁵]')
    plt.savefig(f'freq_response{Order}.png')
    plt.clf()

    # Plot the phase of the model / samples
    plt.plot(omega_model / (2 * np.pi), 180 / np.pi * np.angle(np.squeeze(Z_model)), 'r-.', linewidth=1.5)
    plt.plot(omega / (2 * np.pi), 180 / np.pi * np.angle(np.squeeze(Z)), 'bo')
    plt.legend([f'Model (order={Order})', 'Samples'], loc='lower right')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [deg]')
    plt.savefig(f'phase_response{Order}.png')
    plt.clf()

