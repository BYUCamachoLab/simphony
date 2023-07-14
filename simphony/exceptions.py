"""Simphony exceptions."""


class SimphonyError(Exception):
    """Base error for all simphony errors."""

    pass


class ModelValidationError(SimphonyError):
    """Error raised when a component model is improperly defined."""

    pass


class ConnectionError(SimphonyError):
    """Error raised when an error occurs during component connection."""

    pass


class ShapeMismatchError(SimphonyError):
    """Error raised when the shape of an array is incorrect."""

    pass


class SeparatedCircuitError(SimphonyError):
    """Error raised when a circuit contains two subcircuits that are not
    internally connected."""

    pass
