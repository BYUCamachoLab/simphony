from copy import deepcopy
from types import SimpleNamespace


class Context(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def export(self) -> dict:
        """
        Simulations should export the context to a dictionary before
        modifying it. This allows the context to be restored to its
        original state after the simulation is complete.
        """
        return deepcopy(self.__dict__)
    
    def load(self, d):
        """
        Reload the context from the dictionary returned by export().
        """
        self.__dict__.update(d)


CTX = Context()

# Effective index
CTX.neff = 2.34
CTX.ng = 3.4

# Temperature in C 
CTX.TC = 30 

# Rectangular or polar coordinates
CTX.form = "rect" or "polar" 

# Loss in dB/cm
CTX.loss_db_cm = 1.5
