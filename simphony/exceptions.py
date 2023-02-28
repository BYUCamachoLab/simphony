"""
Simphony exceptions.
"""


class SimphonyError(Exception):
    """
    Base error for all simphony errors.
    """
    pass


class ModelValidationError(SimphonyError):
    """
    Error raised when a simulation model is improperly defined.
    """
    pass


class ConnectionError(SimphonyError):
    """
    Error raised when an error occurs during component connection.
    """
    pass