
class Error(Exception):
    """Base class for simphony errors."""
    pass

class DuplicateModelError(Error):
    def __init__(self, component_type):
        """Error for a duplicate component type.

        Parameters
        ----------
        component_type : UniqueModelName
            The object containing the name of the model that is a duplicate
        errors : *
            Any errors corresponding to this error.
        """
        super().__init__("Component with name \'" + component_type + "\' already exists. Can you rename the component?")



class MalformedModelError(Error):
    """Models are checked upon registration with core. If something is amiss,
    this type of error is raised."""
    def __init__(self, model_name, fix):
        super().__init__("User-defined class \'" + model_name + "\' is malformed (" + fix + ')')

class PortError(MalformedModelError):
    """ComponentModel's default port count is 0; however, a real device cannot
    have 0 ports."""
    def __init__(self, model_name):
        super().__init__(model_name, '\'ports\' must be a positive integer')

class CachableParametersError(MalformedModelError):
    """If a model is cachable, s_parameters must be a properly formed tuple
    (size 2) with lists or ndarrays as their elements."""
    def __init__(self, model_name):
        super().__init__(model_name, '\'s_parameters\' for a cachable model must be a 2-element tuple of numpy ndarrays')

class UncachableParametersError(MalformedModelError):
    """If a model is not cachable, s_parameters must be a callable function
    that generates parameters given certain arguments."""
    def __init__(self, model_name):
        super().__init__(model_name, '\'s_parameters\' for an uncachable model must be a callable function that can calculate parameters given n arguments')