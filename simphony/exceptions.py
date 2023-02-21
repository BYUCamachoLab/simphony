"""
Simphony exceptions.
"""


class SimphonyError(Exception):
    pass


class ModelValidationError(SimphonyError):
    """
    Error raised when a simulation model is improperly defined.
    """
    def __init__(self, message):
        super().__init__(message)
