
from simphony.core.base import *
from simphony.core.netlist import *
from simphony.core import connect

import numpy as np

from simphony.errors import *

models = {}

def register_component_model(cls):
    """Registers a component model with simphony.core.
    
    This allows the program to track models and prevent duplicate names. 
    This also allows all imported models to be accessible throughout the 
    lifetime of the program simply by referencing their class name.
    When classes are registered, simple error checking is performed to ensure
    all required properties are present and that the model will not cause
    problems later in the simulation.

    This function is usually used as a decorator to a model class.

    Parameters
    ----------
    cls : class
        The component model class to be registered.
    """
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
    """Deregisters a component model from simphony.core.

    This method is provided in case a model was created but needs to be
    updated, overridden, or deleted. For example, if two models have
    the same name, the one you don't want to use should be deregistered (since
    registered model names must be unique).
    """
    models.pop(name)