
from .base import *
from .netlist import *
from . import connect
from simphony.errors import DuplicateModelError

models = {}

def register_component_model(cls):
    if cls.__name__ not in models.keys():
        models[cls.__name__] = cls
    else:
        raise DuplicateModelError(cls.__name__)
    return cls
        