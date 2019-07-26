
from .base import *
from .netlist import *
from . import connect

import numpy as np

from simphony.errors import *

models = {}

def register_component_model(cls):
    # Check the definition of ports
    if not cls.ports > 0:
        raise PortError(cls.__name__)

    # Check the cachable condition, as it relates to the s-parameters.
    if cls.cachable:
        # s_parameters is not a tuple
        if (not isinstance(cls.s_parameters, (tuple))):
            raise CachableParametersError(cls.__name__)
        # s_parameters is not a tuple of lists or numpy arrays
        if (not isinstance(cls.s_parameters[0], (list, np.ndarray)) and
            not isinstance(cls.s_parameters[1], (list, np.ndarray))):
            raise CachableParametersError(cls.__name__)
    # Model is not cachable. s_parameters must be a callable function able
    # to calculate parameters given arguments.
    else:
        if not callable(cls.s_parameters):
            raise UncachableParametersError(cls.__name__)

    # Register the model
    if cls.__name__ not in models.keys():
        models[cls.__name__] = cls
    else:
        raise DuplicateModelError(cls.__name__)
    return cls
        
def deregister_component_model(name):
    models.pop(name)