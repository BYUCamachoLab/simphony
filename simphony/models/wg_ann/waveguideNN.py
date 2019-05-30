from keras import backend as K
from keras.models import load_model
import numpy as np


def r2Metric(y_true, y_pred):
    '''
    Used to gauge the effectiveness of the ANN during training. While this
    function is literally never used after training, we are required to provide
    it since we trained the keras model with it. In other words, just ignore
    this.
    '''
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def loadWaveguideNN(modelName = 'NN_SiO2_neff.h5'):
    '''
    Used to load the waveguide model into the current python session. This only
    needs to be done once, thankfully, since it does take a few seconds to load.
    Once it is loaded, the "model" can be passed to other functions for
    predictions.

    Input parameters:
    modelName ................. (string) the name of the keras model. The
                                default name (i.e. when no parameter is passed)
                                is NN_SiO2_neff.h5. In case a new keras model
                                is provided, the new name can be passed.
    '''
    return load_model(modelName,custom_objects={'r2Metric': r2Metric})

def getWaveguideIndex(model,wavelength,width,thickness,mode):
    '''
    Used to predict new effective index values using the provided keras model.

    Input parameters:
    model .................... the keras model stored in memory
    wavelength ............... (Nx1) array of N wavelength points of interest
    width .................... (float) the width of the waveguide
    thickness ................ (float) the thickness of the waveguide
    mode ..................... (int) the desired mode with the following code:
                                0 - TE0, 1 - TE1, 2 - TM0, 3 - TM1

    Output parameters:
    Neff ..................... (Nx1) array of effective index values
                               corresponding the waveguide dimensions and N
                               wavelength points.
    '''

    # First, sanitize the input
    wavelength = np.squeeze(wavelength)
    width      = float(width)
    thickness  = float(thickness)
    mode       = int(mode)

    # Next, get the input statistics
    numWavelength = wavelength.size

    # Now we'll consolidate the input
    widthBank     = np.tile(width,(numWavelength,))
    thicknessBank = np.tile(thickness,(numWavelength,))
    X = np.vstack((wavelength,widthBank,thicknessBank)).T

    # Finally we'll predict
    Y = model.predict(X)
    Neff = Y[:,mode]

    # Return the desired mode
    return Neff

if __name__ == "__main__":
    '''
    If we run this file as a script (instead of a module) then the following
    test routine should run. We simply find the TE0 and TM0 effective indices
    for a simple silicon photonic waveguide (500 nm x 220 nm). This can be used
    as a template for calling the above blackbox functions in any context, so
    long as the required libraries are installed and the NN file is in the same
    directory.
    '''
    wavelengths = np.linspace(1.5,1.6,10)
    width = 0.5
    thickness = 0.22
    mode = 0

    model = loadWaveguideNN()
    TE = getWaveguideIndex(model,wavelengths,width,thickness,mode)
    print('TE')
    print(TE)

    mode = 2
    TM = getWaveguideIndex(model,wavelengths,width,thickness,mode)
    print('TM')
    print(TM)
