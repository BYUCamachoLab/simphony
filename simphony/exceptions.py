"""Simphony exceptions."""


class SimphonyError(Exception):
    """Base error for all simphony errors."""


class ModelValidationError(SimphonyError):
    """Error raised when a component model is improperly defined."""


class ShapeMismatchError(SimphonyError):
    """Error raised when the shape of an array is incorrect."""
